#!/usr/bin/env python3
# Enhanced COPPER View to API Mapper

"""
This tool integrates with Confluence and Bitbucket repositories to:
1. Fetch and process documentation from multiple Confluence spaces
2. Retrieve SQL view definitions from Bitbucket
3. Parse SQL queries to extract table and column information
4. Map view attributes to API endpoints and attributes
5. Use Gemini AI to generate comprehensive mapping documentation
6. Answer common questions about the COPPER API
"""

import os
import sys
import json
import re
import time
import logging
import concurrent.futures
import threading
import sqlparse
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from vertexai.generative_models import GenerationConfig, GenerativeModel
import vertexai

# Suppress InsecureRequestWarning for the required insecure connections
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("copper_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CopperAssistant")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dev-cop-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-exp-83.25")

# Confluence Configuration
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://mygroup.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "user@example.com")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "ATATT3xFfGF0YBUowOcWvB7WZKmO6VQaYGJJXrHDVFUZgbj4l4LK4NCdGxAl15MrVvJcvZX3M9-1016XY")
CONFLUENCE_SPACES = ["CE", "itsrch"]  # Multiple spaces to fetch data from

# Bitbucket Configuration
BITBUCKET_URL = os.environ.get("BITBUCKET_URL", "https://cmestash.chicago.cme.com")
BITBUCKET_API_TOKEN = os.environ.get("BITBUCKET_API_TOKEN", "")
PROJECT_KEY = os.environ.get("PROJECT_KEY", "BLRHACKATHON")
REPO_SLUG = os.environ.get("REPO_SLUG", "copper_views")
DIRECTORY_PATH = os.environ.get("DIRECTORY_PATH", "queries")

# Performance settings
MAX_WORKERS = 5
CACHE_SIZE = 1000

# Important API Documentation Pages - direct IDs for quicker access
IMPORTANT_PAGE_IDS = {
    "copper_intro": "224622013",
    "copper_faq": "168711190",
    "view_to_api_mapping": "168617692",
    "copper_quickstart": "168687143",
    "api_endpoints": "168370805",
    "copper_landing": "168508889",
    "supported_operators": "168665138"
}

class ContentExtractor:
    """
    Extract and process content from Confluence HTML, with enhanced table and image processing.
    """
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def extract_content_from_html(html_content, title=""):
        """
        Extract text, tables, and image contexts from HTML content with improved parsing.
        
        Args:
            html_content: The HTML content to process
            title: The title of the content
            
        Returns:
            Dict containing processed text, tables, and image references
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract regular text with context
            text_content = []
            
            # Process headings to maintain document structure
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_level = int(heading.name[1])
                heading_text = heading.text.strip()
                if heading_text:
                    text_content.append(f"{'#' * heading_level} {heading_text}")
            
            # Process paragraphs and list items
            for p in soup.find_all(['p', 'li']):
                if p.text.strip():
                    text_content.append(p.text.strip())
            
            # Extract tables with enhanced processing
            tables = []
            for i, table in enumerate(soup.find_all('table')):
                table_data = []
                
                # Get table title/caption if available
                table_title = table.find('caption')
                table_title = table_title.text.strip() if table_title else f"Table [{i+1}]"
                
                # Get headers
                headers = []
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
                
                # If no headers in thead, try getting from first row
                if not headers:
                    first_row = table.find('tr')
                    if first_row:
                        first_row_cells = first_row.find_all(['th', 'td'])
                        if first_row.find('th') or all(cell.get('class', '') and 'header' in str(cell.get('class', '[]')) for cell in first_row_cells):
                            headers = [th.text.strip() for th in first_row.find_all(['th', 'td'])]
                
                # Process rows
                rows = []
                tbody = table.find('tbody')
                if tbody:
                    for tr in tbody.find_all('tr'):
                        row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                        if any(cell for cell in row):  # Skip empty rows
                            rows.append(row)
                else:
                    # If no tbody, process all rows (skipping the header if we extracted it)
                    all_rows = table.find_all('tr')
                    start_idx = 1 if headers and len(all_rows) > 0 else 0
                    for tr in all_rows[start_idx:]:
                        row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                        if any(cell for cell in row):  # Skip empty rows
                            rows.append(row)
                
                # Convert table to text representation with improved formatting
                table_text = f"**TABLE: {table_title}**\n"
                
                # Format with consistent column widths for better readability
                if headers and rows:
                    # Calculate column widths
                    col_widths = [len(h) for h in headers]
                    for row in rows:
                        for i, cell in enumerate(row):
                            if i < len(col_widths):
                                col_widths[i] = max(col_widths[i], len(str(cell)))
                
                    # Format header row
                    header_row = "| " + " | ".join([h.ljust(col_widths[i]) for i, h in enumerate(headers)]) + " |"
                    separator = "|" + "".join(["-" * (col_widths[i] + 2) + "|" for i, h in enumerate(headers)])
                    table_text += header_row + "\n" + separator + "\n"
                    
                    # Format data rows
                    for row in rows:
                        if len(row) < len(headers):
                            row.extend([""] * (len(headers) - len(row)))
                        row_text = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row) if i < len(col_widths)]) + " |"
                        table_text += row_text + "\n"
                elif rows:  # Table with no headers
                    # Calculate column width
                    max_cols = max(len(row) for row in rows)
                    col_widths = [0] * max_cols
                    for row in rows:
                        for i, cell in enumerate(row):
                            if i < max_cols:
                                col_widths[i] = max(col_widths[i], len(str(cell)))
                    
                    # Format rows
                    for row in rows:
                        if len(row) < max_cols:
                            row.extend([""] * (max_cols - len(row)))
                        row_text = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row) if i < len(col_widths)]) + " |"
                        table_text += row_text + "\n"
                
                if table_text.strip() != f"**TABLE: {table_title}**":  # Only add tables with content
                    tables.append(table_text)
                    
                    # Also extract table data in a structured format for easier processing
                    structured_table = {
                        "title": table_title,
                        "headers": headers,
                        "rows": rows
                    }
                    table_data.append(structured_table)
            
            # Extract images with improved context
            images = []
            for img in soup.find_all('img'):
                # Get image attributes
                alt_text = img.get('alt', '').strip()
                title_text = img.get('title', '').strip()
                src = img.get('src', '')
                
                # Try to get contextual information
                context = ""
                # Check parent elements for figure captions
                parent_fig = img.find_parent('figure')
                if parent_fig:
                    fig_caption = parent_fig.find('figcaption')
                    if fig_caption:
                        context = fig_caption.text.strip()
                
                # If no caption found, try to get surrounding text
                if not context:
                    prev_elems = list(img.previous_siblings)
                    next_elems = list(img.next_siblings)
                    prev_text = ""
                    for elem in prev_elems:
                        if hasattr(elem, 'text') and elem.text.strip():
                            prev_text = elem.text.strip()
                            break
                    
                    if prev_text and len(prev_text) < 200:  # Only short contexts
                        context = f"Previous content: {prev_text}"
                
                # Construct meaningful image description
                desc = alt_text or title_text or "Image"
                if context:
                    desc += f" - [{context}]"
                
                images.append(f"[IMAGE: {desc}]")
            
            # Extract code blocks with improved formatting
            code_blocks = []
            for pre in soup.find_all('pre'):
                code = pre.find('code')
                if code:
                    # Check for any language specification
                    code_class = code.get('class', [])
                    lang = ""
                    for cls in code_class:
                        if isinstance(cls, str) and cls.startswith('language-'):
                            lang = cls.replace('language-', '')
                            break
                            
                    code_content = code.text.strip()
                    if lang:
                        code_blocks.append(f"```{lang}\n{code_content}\n```")
                    else:
                        code_blocks.append(f"```\n{code_content}\n```")
                else:
                    # Pre without code tag
                    code_blocks.append(f"```\n{pre.text.strip()}\n```")
            
            # Extract any important structured content
            structured_content = []
            for div in soup.find_all('div'):
                # Look for common Confluence structured content classes
                class_value = div.get('class', [])
                if not isinstance(class_value, list):
                    class_value = [str(class_value)]
                
                classnames = {'panel', 'info', 'note', 'warning', 'callout', 'aui-message'}
                
                if any(class_name in classnames for class_name in class_value):
                    title_elem = div.find(['h1', 'h2', 'h3', 'h4', 'h5'])
                    title_text = title_elem.text.strip() if title_elem else "NOTE"
                    content = div.text.strip()
                    structured_content.append(f"**{title_text}** -- {content}")
            
            return {
                "text": "\n\n".join(text_content),
                "tables": tables,
                "table_data": table_data if 'table_data' in locals() else [],
                "images": images,
                "code_blocks": code_blocks,
                "structured_content": structured_content
            }
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            # Return minimal structure with error message
            return {
                "text": f"Error extracting content: {str(e)}",
                "tables": [],
                "table_data": [],
                "images": [],
                "code_blocks": [],
                "structured_content": []
            }
    
    @staticmethod
    def format_for_context(extracted_content, title=""):
        """
        Format the extracted content for use as context.
        
        Args:
            extracted_content: The dictionary of extracted content
            title: The title of the page
            
        Returns:
            Formatted string containing all the content
        """
        sections = []
        
        if title:
            sections.append(f"# {title}")
        
        if extracted_content.get("text"):
            sections.append(extracted_content["text"])
        
        if extracted_content.get("tables"):
            sections.append("\n\n### Tables:\n")
            sections.extend(extracted_content["tables"])
        
        if extracted_content.get("code_blocks"):
            sections.append("\n\n### Code Examples:\n")
            sections.extend(extracted_content["code_blocks"])
        
        if extracted_content.get("structured_content"):
            sections.append("\n\n### Important Notes:\n")
            sections.extend(extracted_content["structured_content"])
        
        if extracted_content.get("images"):
            sections.append("\n\n### Image Information:\n")
            sections.extend(extracted_content["images"])
        
        return "\n\n".join(sections)

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling and caching."""
    
    def __init__(self, base_url, username, api_token, spaces=None):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance
            username: The username for authentication
            api_token: The API token for authentication
            spaces: List of space keys to fetch content from
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "COPPER AI Python Agent"
        }
        self.session = requests.Session()
        self.timeout = 30
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.spaces = spaces or []
        self.space_pages = {}  # Dictionary to store pages for each space
        
        logger.info(f"Initialized Confluence client for {self.base_url} with spaces: {self.spaces}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence API...")
            response = self.session.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                timeout=self.timeout,
                verify=False
            )
            
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info("Connection to Confluence successful!")
                return True
            else:
                logger.warning("Empty response received during connection test")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    @lru_cache(maxsize=CACHE_SIZE)
    def get_cached_request(self, url, params_str):
        """Cached version of GET requests to reduce API calls."""
        try:
            params = json.loads(params_str)
            response = self.session.get(
                url,
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
                verify=False
            )
            
            response.raise_for_status()
            return response.text
            
        except Exception as e:
            logger.error(f"Error in cached request to {url}: {str(e)}")
            return None
    
    def get_all_pages_in_space(self, space_key, batch_size=100):
        """
        Get all pages in a Confluence space using efficient pagination.
        
        Args:
            space_key: The space key to get all pages from
            batch_size: Number of results per request (max 100)
            
        Returns:
            List of page objects with basic information
        """
        all_pages = []
        start = 0
        has_more = True
        
        logger.info(f"Fetching all pages from space: {space_key}")
        
        # Check if we have cached results
        cache_path = f"cache_{space_key}_pages.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('space_key') == space_key:
                        logger.info(f"Using cached page list from {cache_path}")
                        return cached_data.get('pages', [])
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        # If no cache hit, fetch all pages
        while has_more:
            logger.info(f"Fetching pages batch from start={start}")
            try:
                params = {
                    "spaceKey": space_key,
                    "expand": "history",  # Include basic history info to get last updated date
                    "limit": batch_size,
                    "start": start
                }
                
                # Convert params to string for cache key
                params_str = json.dumps(params, sort_keys=True)
                
                # Try to get from cache first
                response_text = self.get_cached_request(f"{self.api_url}/content", params_str)
                
                if not response_text:
                    logger.warning(f"Empty response when fetching pages at start={start}")
                    break
                
                response_data = json.loads(response_text)
                results = response_data.get('results', [])
                all_pages.extend(results)
                
                # Check if there are more pages
                if "size" in response_data and "limit" in response_data:
                    if response_data["size"] < response_data["limit"]:
                        has_more = False
                    else:
                        start += batch_size
                else:
                    has_more = False
                
                logger.info(f"Fetched {len(results)} pages, total so far: {len(all_pages)} pages")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching pages: {str(e)}")
                break
        
        # Cache the results
        try:
            with open(cache_path, 'w') as f:
                json.dump({"space_key": space_key, "pages": all_pages}, f)
                logger.info(f"Cached {len(all_pages)} pages to {cache_path}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
        
        logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
        return all_pages
    
    def load_all_spaces(self):
        """Load pages from all configured spaces."""
        for space_key in self.spaces:
            self.space_pages[space_key] = self.get_all_pages_in_space(space_key)
            logger.info(f"Loaded {len(self.space_pages[space_key])} pages from space {space_key}")
    
    def get_page_by_id(self, page_id, expand=None):
        """Get a specific page by ID with optional expansion parameters."""
        try:
            params = {}
            if expand:
                params["expand"] = expand
                
            # Convert params to string for cache key
            params_str = json.dumps(params, sort_keys=True)
            
            # Try to get from cache first
            response_text = self.get_cached_request(f"{self.api_url}/content/{page_id}", params_str)
            
            if not response_text:
                logger.warning(f"Empty response received when retrieving page ID: {page_id}")
                return None
                
            content = json.loads(response_text)
            logger.info(f"Successfully retrieved page: {content.get('title', 'Unknown title')}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting page by ID {page_id}: {str(e)}")
            return None
    
    def get_page_content(self, page_id, expand=None):
        """
        Get the content of a page and process it for NLP.
        
        Args:
            page_id: The ID of the page
            expand: What to expand in the page content
            
        Returns:
            Dict containing processed content
        """
        try:
            # Use cached version if available
            cache_key = f"content_{page_id}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            # Otherwise fetch the content
            page = self.get_page_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
            
            # Extract basic metadata
            space_path = page.get('_expandable', {}).get('space', '').split('/')[-1] if '_expandable' in page else ""
            
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/spaces/{space_path}/pages/{page.get('id')}",
                "labels": [label.get('name') for label in page.get('metadata', {}).get('labels', {}).get('results', [])]
            }
            
            # Get raw content
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process with our advanced content extractor
            extracted_content = ContentExtractor.extract_content_from_html(html_content, page.get("title", ""))
            formatted_content = ContentExtractor.format_for_context(extracted_content, page.get("title", ""))
            
            result = {
                "metadata": metadata,
                "content": extracted_content,
                "formatted_content": formatted_content,
                "raw_html": html_content
            }
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def search_content(self, query, spaces=None):
        """
        Search for content across multiple spaces.
        
        Args:
            query: The search query
            spaces: List of space keys to search in (defaults to all configured spaces)
            
        Returns:
            List of relevant pages
        """
        spaces_to_search = spaces or self.spaces
        
        # Ensure we have pages loaded for each space
        for space_key in spaces_to_search:
            if space_key not in self.space_pages:
                self.space_pages[space_key] = self.get_all_pages_in_space(space_key)
        
        all_candidates = []
        
        # Score and select candidate pages from each space
        for space_key in spaces_to_search:
            space_pages = self.space_pages.get(space_key, [])
            
            if not space_pages:
                logger.warning(f"No pages found in space {space_key}")
                continue
                
            # Initial filtering based on title
            query_words = query.lower().split()
            candidates = []
            
            # Domain-specific terms to help with relevance
            domain_terms = ["database", "view", "api", "endpoint", "rest", "mapping", 
                            "copper", "table", "column", "field", "attribute"]
            
            # Score pages based on title matches
            for page in space_pages:
                title = page.get('title', '').lower()
                # Check match score in title
                title_score = 0
                for word in query_words:
                    if word in title:
                        title_score += 3  # Higher weight for title matches
                        
                # If title has matches or contains domain terms, include as candidate
                if title_score > 0 or any(term in title for term in domain_terms):
                    candidates.append((page, title_score, space_key))
            
            all_candidates.extend(candidates)
        
        # Sort all candidates and take top N for detailed analysis
        top_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:20]
        
        logger.info(f"Selected {len(top_candidates)} candidate pages for detailed analysis")
        
        # Fetch content for candidate pages in parallel
        page_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.get_page_content, candidate[0]['id']): candidate for candidate in top_candidates}
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                try:
                    content = future.result()
                    if content:
                        page_data.append((candidate[0], content, candidate[1], candidate[2]))
                except Exception as e:
                    logger.error(f"Error fetching content for page {candidate[0].get('id')}: {str(e)}")
        
        # If no candidates with content, get some recent pages
        if not page_data:
            recent_pages = []
            for space_key in spaces_to_search:
                space_pages = self.space_pages.get(space_key, [])
                if space_pages:
                    # Sort by last updated date
                    recent = sorted(space_pages, 
                                   key=lambda x: x.get('history', {}).get('lastUpdated', '2000-01-01'),
                                   reverse=True)[:5]
                    for page in recent:
                        recent_pages.append((page, 0, space_key))
            
            # Fetch content for these pages
            for page, score, space_key in recent_pages:
                content = self.get_page_content(page['id'])
                if content:
                    page_data.append((page, content, score, space_key))
        
        # Detailed relevance scoring with content
        scored_pages = []
        for page, content, initial_score, space_key in page_data:
            page_text = content['content'].get('text', '')
            
            # Calculate a relevance score
            score = initial_score
            
            # Score based on query word frequency
            for word in query_words:
                word_count = page_text.lower().count(word)
                score += word_count * 0.1  # Add score per occurrence
                
            # Bonus for exact phrase matches
            if len(query_words) > 1:
                phrase_matches = page_text.lower().count(query.lower())
                score += phrase_matches * 2  # Double bonus for exact phrase match
                
            # Specific domain terms bonus
            for term in domain_terms:
                if term in page_text.lower():
                    score += 0.5  # Bonus for each matched domain term
                    
            scored_pages.append((page, content, score, space_key))
            
        # Sort by score and take top results
        top_results = sorted(scored_pages, key=lambda x: x[2], reverse=True)[:5]
        
        logger.info(f"Selected {len(top_results)} most relevant pages")
        
        return top_results

class StashClient:
    """Client for interacting with Bitbucket (Stash) repositories to fetch SQL view definitions."""
    
    def __init__(self, base_url, api_token=None, project_key=None, repo_slug=None, directory_path=None):
        """
        Initialize the Stash client.
        
        Args:
            base_url: The base URL of the Bitbucket instance
            api_token: The API token for authentication
            project_key: The project key 
            repo_slug: The repository slug
            directory_path: The directory path within the repository
        """
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.project_key = project_key
        self.repo_slug = repo_slug
        self.directory_path = directory_path
        self.headers = {
            "Accept": "application/json"
        }
        
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
        
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info(f"Initialized Stash client for {self.base_url}")
    
    def get_file_content(self, filename):
        """
        Get the content of a file from the repository.
        
        Args:
            filename: The name of the file to fetch
            
        Returns:
            The content of the file as a string
        """
        if not filename:
            logger.error("Filename cannot be empty")
            return None
        
        # Check cache first
        cache_key = f"file_{filename}"
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        try:
            file_path_in_repo = f"{self.directory_path}/{filename}"
            url_path = f"rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}/raw/{file_path_in_repo}"
            
            logger.info(f"Fetching file: {file_path_in_repo}")
            
            response = requests.get(
                f"{self.base_url}/{url_path}",
                headers=self.headers,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                content = response.text
                # Cache the result
                with self.cache_lock:
                    self.cache[cache_key] = content
                return content
            elif response.status_code == 401:
                logger.error("Authentication failed. Check your API token.")
                return None
            elif response.status_code == 404:
                logger.error(f"File not found: {filename}")
                return None
            else:
                logger.error(f"Error fetching file: Status code {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching file {filename}: {str(e)}")
            return None
    
    def list_files(self):
        """
        List all files in the configured directory.
        
        Returns:
            List of filenames
        """
        try:
            url_path = f"rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}/files/{self.directory_path}"
            
            logger.info(f"Listing files in {self.directory_path}")
            
            response = requests.get(
                f"{self.base_url}/{url_path}",
                headers=self.headers,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                files = response.json()
                return [f for f in files if f.endswith('.sql')]
            else:
                logger.error(f"Error listing files: Status code {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    def search_file_by_view_name(self, view_name):
        """
        Search for a file containing the specified view name.
        
        Args:
            view_name: The name of the view to search for
            
        Returns:
            The filename if found, None otherwise
        """
        # Normalize view name for comparison
        view_name = view_name.lower().strip()
        
        # List all SQL files
        sql_files = self.list_files()
        
        for filename in sql_files:
            # Check if view name is part of filename
            if view_name in filename.lower():
                return filename
        
        # If not found by filename, we could check file contents
        for filename in sql_files:
            content = self.get_file_content(filename)
            if content and f"create view {view_name}" in content.lower():
                return filename
        
        logger.warning(f"No file found for view: {view_name}")
        return None

class SQLParser:
    """Parser for SQL statements to extract table and column information."""
    
    @staticmethod
    def parse_sql(sql_text):
        """
        Parse SQL text to extract important information.
        
        Args:
            sql_text: The SQL text to parse
            
        Returns:
            Dict containing extracted information
        """
        if not sql_text:
            return {
                "view_name": None,
                "columns": [],
                "tables": [],
                "joins": [],
                "conditions": []
            }
        
        # Normalize SQL (convert to lowercase, normalize whitespace)
        normalized_sql = re.sub(r'\s+', ' ', sql_text).strip()
        
        # Extract view name
        view_name_match = re.search(r'create\s+(?:or\s+replace\s+)?view\s+([a-zA-Z0-9_.\[\]"]+)', normalized_sql, re.IGNORECASE)
        view_name = view_name_match.group(1) if view_name_match else None
        
        # Remove quoted strings to make parsing easier
        sql_no_strings = re.sub(r"'[^']*'", "''", normalized_sql)
        
        # Extract columns from SELECT clause
        columns = []
        select_match = re.search(r'select\s+(.*?)\s+from', sql_no_strings, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            
            # Handle cases with * (select all columns)
            if '*' in select_clause:
                columns.append({'name': '*', 'alias': None, 'source': None})
            else:
                # Split by commas but be careful of functions
                # This is a simple approach and won't handle all cases
                parenthesis_level = 0
                current_column = ""
                for char in select_clause:
                    if char == '(':
                        parenthesis_level += 1
                    elif char == ')':
                        parenthesis_level -= 1
                    
                    if char == ',' and parenthesis_level == 0:
                        # End of a column
                        columns.append(SQLParser.parse_column(current_column.strip()))
                        current_column = ""
                    else:
                        current_column += char
                
                # Add the last column
                if current_column.strip():
                    columns.append(SQLParser.parse_column(current_column.strip()))
        
        # Extract tables from FROM and JOIN clauses
        tables = []
        joins = []
        
        # Extract FROM clause
        from_match = re.search(r'from\s+(.*?)(?:\s+where|\s+group|\s+having|\s+order|\s+limit|\s*$)', 
                              sql_no_strings, re.IGNORECASE | re.DOTALL)
        if from_match:
            from_clause = from_match.group(1)
            
            # Extract tables and joins
            join_pattern = r'(?:inner|outer|left|right|full|cross)?\s*join\s+([a-zA-Z0-9_.\[\]"]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?\s+on\s+(.*?)(?=\s+(?:inner|outer|left|right|full|cross)?\s*join|\s*$)'
            join_matches = re.finditer(join_pattern, from_clause, re.IGNORECASE)
            
            for match in join_matches:
                table_name = match.group(1)
                alias = match.group(2) if match.group(2) else None
                join_condition = match.group(3).strip()
                
                tables.append({
                    'name': table_name,
                    'alias': alias
                })
                
                joins.append({
                    'table': table_name,
                    'alias': alias,
                    'condition': join_condition
                })
            
            # Extract the main table (first in FROM clause)
            main_table_pattern = r'^\s*([a-zA-Z0-9_.\[\]"]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?'
            main_table_match = re.search(main_table_pattern, from_clause, re.IGNORECASE)
            
            if main_table_match:
                table_name = main_table_match.group(1)
                alias = main_table_match.group(2) if main_table_match.group(2) else None
                
                # Add to tables if not already there
                if not any(t['name'] == table_name for t in tables):
                    tables.append({
                        'name': table_name,
                        'alias': alias
                    })
        
        # Extract WHERE conditions
        conditions = []
        where_match = re.search(r'where\s+(.*?)(?:\s+group|\s+having|\s+order|\s+limit|\s*$)', 
                               sql_no_strings, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Simple splitting by AND/OR operators
            # This won't handle complex conditions correctly
            and_conditions = re.split(r'\s+and\s+', where_clause, flags=re.IGNORECASE)
            
            for and_condition in and_conditions:
                or_conditions = re.split(r'\s+or\s+', and_condition, flags=re.IGNORECASE)
                for condition in or_conditions:
                    condition = condition.strip()
                    if condition:
                        conditions.append(condition)
        
        return {
            "view_name": view_name,
            "columns": columns,
            "tables": tables,
            "joins": joins,
            "conditions": conditions,
            "sql_text": sql_text
        }
    
    @staticmethod
    def parse_column(column_expr):
        """
        Parse a column expression from a SELECT clause.
        
        Args:
            column_expr: The column expression to parse
            
        Returns:
            Dict containing column information
        """
        column = {
            'name': column_expr,
            'alias': None,
            'source': None
        }
        
        # Check for AS alias
        as_match = re.search(r'(.*?)\s+as\s+([a-zA-Z0-9_]+)\s*$', column_expr, re.IGNORECASE)
        if as_match:
            column['name'] = as_match.group(1).strip()
            column['alias'] = as_match.group(2).strip()
        
        # Check for implicit alias (no AS keyword)
        elif re.search(r'(.*?)\s+([a-zA-Z0-9_]+)\s*$', column_expr):
            match = re.search(r'(.*?)\s+([a-zA-Z0-9_]+)\s*$', column_expr)
            # Only consider it an alias if there's a dot or function
            if '.' in match.group(1) or '(' in match.group(1):
                column['name'] = match.group(1).strip()
                column['alias'] = match.group(2).strip()
        
        # Extract source (table) if it's a simple column reference
        if '.' in column['name'] and '(' not in column['name']:
            parts = column['name'].split('.')
            if len(parts) == 2:
                column['source'] = parts[0].strip()
                column['name'] = parts[1].strip()
        
        return column

class MappingGenerator:
    """Generate API mappings from database view attributes."""
    
    def __init__(self, confluence_client, stash_client):
        """
        Initialize the mapping generator.
        
        Args:
            confluence_client: The Confluence client to fetch API documentation
            stash_client: The Stash client to fetch view definitions
        """
        self.confluence = confluence_client
        self.stash = stash_client
        self.mapping_cache = {}
        self.sql_parser = SQLParser()
        
    def generate_mapping(self, view_sql=None, view_name=None):
        """
        Generate a mapping from a view to API endpoints.
        
        Args:
            view_sql: The SQL definition of the view
            view_name: The name of the view (if SQL not provided directly)
            
        Returns:
            Dict containing the mapping
        """
        # Check cache first
        cache_key = view_name or hashlib.md5(view_sql.encode()).hexdigest()
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]
        
        # Get the SQL if only view name is provided
        if not view_sql and view_name:
            filename = self.stash.search_file_by_view_name(view_name)
            if filename:
                view_sql = self.stash.get_file_content(filename)
                if not view_sql:
                    logger.error(f"Could not fetch SQL for view: {view_name}")
                    return None
            else:
                logger.error(f"Could not find file for view: {view_name}")
                return None
        
        # Parse the SQL
        parsed_sql = SQLParser.parse_sql(view_sql)
        if not parsed_sql["view_name"]:
            logger.error("Could not parse view name from SQL")
            return None
        
        # The actual view name (might be different from the provided one)
        actual_view_name = parsed_sql["view_name"]
        
        # Search for relevant API documentation
        search_queries = [
            f"API mapping {actual_view_name}",
            f"{actual_view_name} API endpoint",
            "view to API mapping",
            "COPPER API mapping"
        ]
        
        relevant_pages = []
        for query in search_queries:
            results = self.confluence.search_content(query)
            for result in results:
                if result not in relevant_pages:
                    relevant_pages.append(result)
        
        # Extract table data from relevant pages
        mapping_data = self.extract_mapping_data_from_pages(relevant_pages, parsed_sql)
        
        # If no mapping found from the pages, try to generate it using Gemini AI
        if not mapping_data["attributes"]:
            # We'll implement this in the GeminiAssistant class
            logger.warning(f"No mapping found for view {actual_view_name} in documentation")
            mapping_data["generated"] = True
        
        # Cache the result
        self.mapping_cache[cache_key] = mapping_data
        
        return mapping_data
    
    def extract_mapping_data_from_pages(self, pages, parsed_sql):
        """
        Extract mapping data from Confluence pages.
        
        Args:
            pages: List of relevant pages
            parsed_sql: Parsed SQL information
            
        Returns:
            Dict containing extracted mapping data
        """
        view_name = parsed_sql["view_name"]
        mapping_data = {
            "view_name": view_name,
            "attributes": [],
            "api_endpoints": [],
            "notes": [],
            "generated": False
        }
        
        for page, content, score, space_key in pages:
            # Extract mapping from tables
            table_data = content["content"].get("table_data", [])
            
            for table in table_data:
                # Check if this looks like a mapping table
                headers = table.get("headers", [])
                headers_lower = [h.lower() for h in headers]
                
                # Look for view-to-API mapping patterns in headers
                view_headers = ["view", "view attribute", "view column", "copper view", "db view", "field"]
                api_headers = ["api", "api endpoint", "api attribute", "endpoint", "copper api", "rest api"]
                
                view_col_idx = -1
                api_col_idx = -1
                
                for i, header in enumerate(headers_lower):
                    if any(term in header for term in view_headers):
                        view_col_idx = i
                    if any(term in header for term in api_headers):
                        api_col_idx = i
                
                # If we found both view and API columns
                if view_col_idx >= 0 and api_col_idx >= 0:
                    for row in table.get("rows", []):
                        if len(row) > max(view_col_idx, api_col_idx):
                            view_attr = row[view_col_idx].strip()
                            api_attr = row[api_col_idx].strip()
                            
                            # Extract notes if there are more columns
                            notes = []
                            for i, cell in enumerate(row):
                                if i != view_col_idx and i != api_col_idx and cell.strip():
                                    # If we have a header for this column, use it
                                    if i < len(headers):
                                        notes.append(f"{headers[i]}: {cell.strip()}")
                                    else:
                                        notes.append(cell.strip())
                            
                            # Only add if this looks like a valid mapping
                            if view_attr and api_attr:
                                mapping_data["attributes"].append({
                                    "view_attribute": view_attr,
                                    "api_attribute": api_attr,
                                    "notes": notes
                                })
                
                # Also look for API endpoint information
                endpoint_headers = ["endpoint", "api endpoint", "copper api", "rest api"]
                endpoint_idx = -1
                
                for i, header in enumerate(headers_lower):
                    if any(term in header for term in endpoint_headers):
                        endpoint_idx = i
                        break
                
                if endpoint_idx >= 0:
                    for row in table.get("rows", []):
                        if len(row) > endpoint_idx:
                            endpoint = row[endpoint_idx].strip()
                            if endpoint and "/v" in endpoint:  # Simple check for API endpoint format
                                if endpoint not in [e["endpoint"] for e in mapping_data["api_endpoints"]]:
                                    endpoint_data = {"endpoint": endpoint, "description": ""}
                                    
                                    # Look for a description column
                                    for i, header in enumerate(headers_lower):
                                        if "description" in header and i < len(row):
                                            endpoint_data["description"] = row[i].strip()
                                            break
                                    
                                    mapping_data["api_endpoints"].append(endpoint_data)
            
            # Extract notes from text content
            text_content = content["content"].get("text", "")
            view_mentions = re.findall(rf'{re.escape(view_name)}\s+.*?(?:\.|$)', text_content, re.IGNORECASE)
            for mention in view_mentions:
                if "api" in mention.lower():
                    mapping_data["notes"].append(mention.strip())
        
        return mapping_data

class GeminiAssistant:
    """Assistant powered by Google Gemini for generating responses and mappings."""
    
    def __init__(self, confluence_client, stash_client):
        """
        Initialize the Gemini Assistant.
        
        Args:
            confluence_client: The Confluence client for fetching documentation
            stash_client: The Stash client for fetching view definitions
        """
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        self.confluence = confluence_client
        self.stash = stash_client
        self.mapping_generator = MappingGenerator(confluence_client, stash_client)
        self.memory = []  # Simple conversation memory
        
        # Cache for generated responses
        self.response_cache = {}
        
        logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")
    
    def answer_question(self, question):
        """
        Answer a question using the Gemini model and COPPER context.
        
        Args:
            question: The user's question
            
        Returns:
            The generated answer
        """
        # Check for specific intent types
        intent = self.detect_intent(question)
        
        if intent == "view_mapping":
            # Extract view name from question
            view_name = self.extract_view_name(question)
            if view_name:
                return self.generate_view_mapping(view_name)
            else:
                return "I couldn't identify which view you want me to map. Please specify the view name."
                
        elif intent == "sql_to_api_mapping":
            # Extract SQL from question if present
            sql = self.extract_sql(question)
            if sql:
                return self.generate_mapping_from_sql(sql)
            else:
                return "I couldn't find a SQL query in your question. Please provide the view definition."
                
        elif intent == "api_info":
            return self.generate_api_info_response(question)
            
        else:
            # General documentation question
            return self.generate_documentation_response(question)
    
    def detect_intent(self, question):
        """
        Detect the intent of the question.
        
        Args:
            question: The user's question
            
        Returns:
            The detected intent
        """
        question_lower = question.lower()
        
        # Check for view mapping intent
        view_mapping_patterns = [
            r"map(?:ping)?\s+(?:for|of|to)?\s+(?:the\s+)?view",
            r"which\s+api\s+(?:for|to\s+use\s+for)\s+(?:the\s+)?view",
            r"how\s+(?:do\s+I|to)\s+map\s+(?:the\s+)?view",
            r"what\s+is\s+the\s+mapping\s+for",
            r"find\s+(?:the\s+)?mapping\s+for"
        ]
        
        for pattern in view_mapping_patterns:
            if re.search(pattern, question_lower):
                return "view_mapping"
        
        # Check for SQL to API mapping intent
        sql_patterns = [
            r"create\s+(?:or\s+replace\s+)?view",
            r"select\s+.*\s+from",
            r"map\s+this\s+sql",
            r"convert\s+this\s+sql",
            r"sql\s+query\s+to\s+api"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, question_lower):
                return "sql_to_api_mapping"
        
        # Check for API information intent
        api_info_patterns = [
            r"what\s+(?:is|are)\s+(?:the\s+)?api",
            r"how\s+(?:do\s+I|to)\s+use\s+(?:the\s+)?api",
            r"api\s+(?:documentation|docs|endpoints|usage)",
            r"explain\s+(?:the\s+)?api",
            r"tell\s+me\s+about\s+(?:the\s+)?api"
        ]
        
        for pattern in api_info_patterns:
            if re.search(pattern, question_lower):
                return "api_info"
        
        # Default to general documentation
        return "general_documentation"
    
    def extract_view_name(self, question):
        """
        Extract the view name from the question.
        
        Args:
            question: The user's question
            
        Returns:
            The extracted view name or None
        """
        # Try to match view name patterns
        view_patterns = [
            r"(?:view|table)\s+(?:called|named)\s+(?P<view>[A-Za-z0-9_]+)",
            r"(?:view|table)\s+(?P<view>[A-Za-z0-9_]+)",
            r"mapping\s+for\s+(?P<view>[A-Za-z0-9_]+)",
            r"(?P<view>[A-Za-z0-9_]+)\s+view",
            r"W_[A-Za-z0-9_]+"  # Common pattern for COPPER views
        ]
        
        for pattern in view_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                # Return the named group if it exists, otherwise the entire match
                return match.group("view") if "view" in match.groupdict() else match.group(0)
        
        return None
    
    def extract_sql(self, question):
        """
        Extract SQL query from the question.
        
        Args:
            question: The user's question
            
        Returns:
            The extracted SQL or None
        """
        # Look for SQL query between backticks or code blocks
        code_block_patterns = [
            r"```sql\s*([\s\S]*?)\s*```",
            r"```\s*(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW[\s\S]*?)\s*```",
            r"```\s*(SELECT[\s\S]*?)\s*```",
            r"`(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW[\s\S]*?)`",
            r"`(SELECT[\s\S]*?)`"
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for SQL keywords
        sql_patterns = [
            r"(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+\w+\s+AS\s+SELECT[\s\S]*?)(;|$|Please)",
            r"(SELECT\s+[\s\S]*?FROM\s+[\s\S]*?)(;|$|Please)"
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def generate_view_mapping(self, view_name):
        """
        Generate a mapping for a specific view.
        
        Args:
            view_name: The name of the view
            
        Returns:
            The generated mapping
        """
        logger.info(f"Generating mapping for view: {view_name}")
        
        # Get the view SQL from Stash
        filename = self.stash.search_file_by_view_name(view_name)
        if not filename:
            return f"I couldn't find the definition for view {view_name}. Please check the view name or provide the SQL definition directly."
        
        sql = self.stash.get_file_content(filename)
        if not sql:
            return f"I found a file for view {view_name}, but couldn't read its content. Please try again or provide the SQL definition directly."
        
        # Generate mapping
        mapping = self.mapping_generator.generate_mapping(view_sql=sql)
        
        if not mapping:
            # If no mapping found, generate using Gemini
            return self.generate_mapping_from_sql(sql)
        
        if mapping.get("generated", False) or not mapping.get("attributes"):
            # If mapping was generated or empty, generate using Gemini
            return self.generate_mapping_from_sql(sql)
        
        # Format mapping results
        result = f"## Mapping for {mapping['view_name']}\n\n"
        
        # Add API endpoints
        if mapping.get("api_endpoints"):
            result += "### API Endpoints\n"
            for endpoint in mapping["api_endpoints"]:
                result += f"- `{endpoint['endpoint']}`"
                if endpoint.get("description"):
                    result += f": {endpoint['description']}"
                result += "\n"
            result += "\n"
        
        # Add attribute mappings
        if mapping.get("attributes"):
            result += "### Attribute Mappings\n"
            result += "| View Attribute | API Attribute | Notes |\n"
            result += "|---------------|--------------|-------|\n"
            
            for attr in mapping["attributes"]:
                notes = " ".join(attr.get("notes", []))
                result += f"| {attr['view_attribute']} | {attr['api_attribute']} | {notes} |\n"
            
            result += "\n"
        
        # Add notes
        if mapping.get("notes"):
            result += "### Additional Notes\n"
            for note in mapping["notes"]:
                result += f"- {note}\n"
        
        return result
    
    def generate_mapping_from_sql(self, sql):
        """
        Generate a mapping directly from SQL using Gemini.
        
        Args:
            sql: The SQL definition of the view
            
        Returns:
            The generated mapping
        """
        logger.info("Generating mapping from SQL using Gemini")
        
        # Parse the SQL to get structure
        parsed_sql = SQLParser.parse_sql(sql)
        
        # Get relevant context from Confluence for the API mapping
        context_pages = self.confluence.search_content("view to API mapping")
        context_text = ""
        
        for page, content, score, space_key in context_pages[:3]:  # Take top 3 most relevant pages
            context_text += content["formatted_content"] + "\n\n"
        
        # Check if we should explicitly look at the mapping page
        mapping_page_id = IMPORTANT_PAGE_IDS.get("view_to_api_mapping")
        if mapping_page_id:
            mapping_page = self.confluence.get_page_content(mapping_page_id)
            if mapping_page:
                context_text += mapping_page["formatted_content"] + "\n\n"
        
        # Build the prompt for Gemini
        prompt = f"""
        You are a COPPER mapping expert. Your task is to create a mapping between a database view and the corresponding COPPER API endpoints and attributes. Here is the SQL definition of the view:
        
        ```sql
        {sql}
        ```
        
        Based on this view definition and the following API documentation, create a structured mapping that shows:
        1. The main API endpoint(s) that correspond to this view
        2. For each column/attribute in the view, the corresponding API attribute
        3. Any additional notes that would help understand the mapping
        
        Here is relevant information about COPPER API mappings:
        
        {context_text}
        
        Please format your response as a structured JSON object with these fields:
        - "view_name": The name of the view
        - "api_endpoints": Array of API endpoints with their descriptions
        - "attribute_mappings": Array of mappings from view attributes to API attributes
        - "notes": Any important notes about the mapping
        
        Provide as much detail as possible based on the available information. If you need to make educated guesses, explain your reasoning.
        """
        
        # Get mappings from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=8000,
            )
        )
        
        # Extract JSON from response
        if response.candidates and response.candidates[0].text:
            response_text = response.candidates[0].text.strip()
            
            # Try to extract JSON
            json_matches = re.findall(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_matches:
                json_text = json_matches[0]
            else:
                # Try to extract between curly braces
                json_matches = re.findall(r'(\{\s*"view_name"[\s\S]*?\})', response_text)
                if json_matches:
                    json_text = json_matches[0]
                else:
                    json_text = response_text
            
            try:
                mapping_data = json.loads(json_text)
                # Convert to formatted text
                return self.format_mapping_results(mapping_data)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the formatted response directly
                return self.format_ai_response(response_text)
        
        return "I couldn't generate a mapping for this SQL. Please check if the SQL is valid or try a different view."
    
    def format_mapping_results(self, mapping_data):
        """
        Format mapping results into a readable format.
        
        Args:
            mapping_data: The mapping data
            
        Returns:
            Formatted text
        """
        result = f"## Mapping for {mapping_data.get('view_name', 'View')}\n\n"
        
        # Add API endpoints
        if mapping_data.get("api_endpoints"):
            result += "### API Endpoints\n"
            for endpoint in mapping_data["api_endpoints"]:
                if isinstance(endpoint, dict):
                    ep = endpoint.get("endpoint", endpoint)
                    desc = endpoint.get("description", "")
                    result += f"- `{ep}`"
                    if desc:
                        result += f": {desc}"
                else:
                    result += f"- `{endpoint}`"
                result += "\n"
            result += "\n"
        
        # Add attribute mappings
        if mapping_data.get("attribute_mappings"):
            result += "### Attribute Mappings\n"
            result += "| View Attribute | API Attribute | Notes |\n"
            result += "|---------------|--------------|-------|\n"
            
            for attr in mapping_data["attribute_mappings"]:
                if isinstance(attr, dict):
                    view_attr = attr.get("view_attribute", attr.get("view_column", ""))
                    api_attr = attr.get("api_attribute", attr.get("api_field", ""))
                    notes = attr.get("notes", attr.get("description", ""))
                    
                    result += f"| {view_attr} | {api_attr} | {notes} |\n"
            
            result += "\n"
        
        # Add notes
        if mapping_data.get("notes"):
            result += "### Additional Notes\n"
            if isinstance(mapping_data["notes"], list):
                for note in mapping_data["notes"]:
                    result += f"- {note}\n"
            else:
                result += mapping_data["notes"] + "\n"
        
        # Add disclaimer if this was AI generated
        result += "\n*Note: This mapping was generated using AI based on the view definition and available documentation. Please verify the accuracy before implementation.*"
        
        return result
    
    def generate_api_info_response(self, question):
        """
        Generate a response about the COPPER API.
        
        Args:
            question: The user's question
            
        Returns:
            The generated response
        """
        logger.info(f"Generating API info response for: {question}")
        
        # Check cache first
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Get content from important API pages
        api_context = ""
        api_page_ids = [
            IMPORTANT_PAGE_IDS.get("copper_intro"),
            IMPORTANT_PAGE_IDS.get("copper_faq"),
            IMPORTANT_PAGE_IDS.get("copper_quickstart"),
            IMPORTANT_PAGE_IDS.get("api_endpoints")
        ]
        
        for page_id in api_page_ids:
            if page_id:
                page = self.confluence.get_page_content(page_id)
                if page:
                    api_context += page["formatted_content"] + "\n\n"
        
        # Also search for specific terms in the question
        search_terms = re.findall(r'\b\w{3,}\b', question.lower())
        for term in search_terms:
            if len(term) >= 3:  # Only search for terms with at least 3 characters
                results = self.confluence.search_content(term)
                for page, content, score, space_key in results[:2]:  # Top 2 results
                    api_context += content["formatted_content"] + "\n\n"
        
        # Build prompt for Gemini
        prompt = f"""
        You are a COPPER API expert. Answer the following question using the COPPER API documentation provided.
        
        User question: {question}
        
        COPPER API documentation:
        {api_context}
        
        Provide a detailed, accurate answer based on the documentation. Include specific API endpoints, parameters, and examples where applicable. If the information is not available in the documentation, acknowledge this and provide the best guidance you can based on the general principles of the COPPER API.
        """
        
        # Generate response
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=4000,
            )
        )
        
        if response.candidates and response.candidates[0].text:
            result = self.format_ai_response(response.candidates[0].text.strip())
            # Cache the result
            self.response_cache[cache_key] = result
            return result
        
        return "I couldn't find specific information about that in the COPPER API documentation. Please try asking in a different way or check the official documentation."
    
    def generate_documentation_response(self, question):
        """
        Generate a response based on COPPER documentation.
        
        Args:
            question: The user's question
            
        Returns:
            The generated response
        """
        logger.info(f"Generating documentation response for: {question}")
        
        # Check cache first
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Search for relevant content
        results = self.confluence.search_content(question)
        context = ""
        
        for page, content, score, space_key in results:
            context += content["formatted_content"] + "\n\n"
        
        # Build prompt for Gemini
        prompt = f"""
        You are a COPPER expert. Answer the following question using the documentation provided.
        
        User question: {question}
        
        COPPER documentation:
        {context}
        
        Previous conversation context:
        {self.get_conversation_context()}
        
        Provide a detailed, accurate answer based on the documentation. Include specific examples or references where applicable. If the information is not available in the documentation, acknowledge this and provide the best guidance you can based on general knowledge of database systems and APIs.
        """
        
        # Generate response
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=4000,
            )
        )
        
        if response.candidates and response.candidates[0].text:
            result = self.format_ai_response(response.candidates[0].text.strip())
            # Cache the result
            self.response_cache[cache_key] = result
            # Update conversation memory
            self.memory.append({"question": question, "answer": result})
            if len(self.memory) > 5:  # Keep only last 5 exchanges
                self.memory.pop(0)
            return result
        
        return "I couldn't find specific information about that in the COPPER documentation. Please try asking in a different way or provide more context."
    
    def get_conversation_context(self):
        """
        Get previous conversation context for continuity.
        
        Returns:
            Formatted conversation context
        """
        if not self.memory:
            return "No previous conversation."
        
        context = "Previous exchanges:\n"
        for i, exchange in enumerate(self.memory[-3:]):  # Last 3 exchanges
            context += f"Q{i+1}: {exchange['question']}\n"
            context += f"A{i+1}: {exchange['answer'][:100]}...\n\n"
        
        return context
    
    def format_ai_response(self, text):
        """
        Format the AI response to make it more readable.
        
        Args:
            text: The raw response text
            
        Returns:
            Formatted response
        """
        # Remove extra AI self-references
        text = re.sub(r'As an AI|As a language model|As a COPPER expert', '', text)
        
        # Ensure code blocks are properly formatted
        text = re.sub(r'```(\w+)\s', r'```\1\n', text)
        
        # Remove redundant disclaimers
        text = re.sub(r'Please note that this information is based on the documentation provided\.', '', text)
        text = re.sub(r'This answer is based on the COPPER documentation provided\.', '', text)
        
        return text.strip()

def main():
    """Main function to run the COPPER View to API Mapper."""
    logger.info("Starting COPPER View to API Mapper")
    
    # Initialize components
    print("Initializing COPPER View to API Mapper...")
    
    # Initialize Confluence client
    confluence = ConfluenceClient(
        CONFLUENCE_URL, 
        CONFLUENCE_USERNAME, 
        CONFLUENCE_API_TOKEN,
        spaces=CONFLUENCE_SPACES
    )
    
    # Initialize Stash client
    stash = StashClient(
        BITBUCKET_URL,
        BITBUCKET_API_TOKEN,
        PROJECT_KEY,
        REPO_SLUG,
        DIRECTORY_PATH
    )
    
    # Test connections
    print("Testing connections...")
    if not confluence.test_connection():
        logger.error("Failed to connect to Confluence. Check your credentials.")
        print("Error: Failed to connect to Confluence. Please check your credentials.")
        return
    
    print("Preloading documentation from Confluence...")
    confluence.load_all_spaces()
    
    # Initialize Gemini Assistant
    assistant = GeminiAssistant(confluence, stash)
    
    print("\n======== COPPER View to API Mapping Assistant ========")
    print(f"Loaded data from {len(CONFLUENCE_SPACES)} Confluence spaces.")
    print("I can help you map COPPER database views to REST APIs, answer questions about the COPPER API,")
    print("and provide information about the mapping process.")
    print("\nAsk me questions like:")
    print("- What is the mapping for view W_CORE_TCC_SPAN_MAPPING?")
    print("- Which API can I use to get product data?")
    print("- How do I find all the instruments for an exchange?")
    print("- What operators are supported for timestamps?")
    
    while True:
        try:
            user_input = input("\nQuestion (type 'exit' to quit): ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Thanks for using the COPPER View to API Mapper. Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\nWorking on your request...")
            start_time = time.time()
            answer = assistant.answer_question(user_input)
            end_time = time.time()
            
            print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
            print("-------------------------------------------------------")
            print(answer)
            
        except KeyboardInterrupt:
            print("\nOperation interrupted. Goodbye!")
            break
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            print(f"Sorry, I ran into an issue: {str(e)}. Let's try a different question.")

if __name__ == "__main__":
    main()
