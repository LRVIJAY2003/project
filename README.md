#!/usr/bin/env python3
# COPPER View to API Mapper - CME BLR Hackathon 2025

"""
Advanced COPPER View to API Mapper
----------------------------------
A robust solution for mapping database views to REST APIs:
1. Extracts and caches all knowledge from Confluence documentation
2. Retrieves SQL view definitions from Bitbucket with reliable fallbacks
3. Accurately maps database views to API endpoints and attributes
4. Generates comprehensive API request/response examples
5. Provides conversational, professional assistance for COPPER API questions
"""

import os
import sys
import json
import re
import time
import logging
import concurrent.futures
import threading
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import hashlib
import sqlparse
from urllib.parse import urljoin

# GenAI imports
from vertexai.generative_models import GenerationConfig, GenerativeModel
import vertexai

# Suppress InsecureRequestWarning
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
MAX_WORKERS = 10
MAX_CONTENT_SIZE = 16000
NO_LIMIT = 10000  # Effectively no limit on document retrieval

# Important documentation pages with direct access IDs
IMPORTANT_PAGES = {
    "copper_intro": {"id": "224622013", "title": "COPPER APP/A"},
    "api_faq": {"id": "168711190", "title": "COPPER API Frequently Asked Questions"},
    "view_to_api_mapping": {"id": "168617692", "title": "View to API Mapping"},
    "api_quickstart": {"id": "168687143", "title": "COPPER API QUICK START GUIDE"},
    "api_endpoints": {"id": "168370805", "title": "API Endpoint"},
    "api_landing": {"id": "168508889", "title": "COPPER API First Landing Page"},
    "supported_operators": {"id": "168665138", "title": "COPPER SQL API Supported Operators"}
}

class ContentExtractor:
    """Extract structured content from Confluence HTML documents."""
    
    @staticmethod
    def extract_content(html_content, title=""):
        """
        Extract structured content from HTML with enhanced parsing for tables, lists, and code blocks.
        
        Args:
            html_content: Raw HTML content
            title: Page title
            
        Returns:
            Dict with extracted content sections
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize result structure
            extracted = {
                "title": title,
                "text_blocks": [],
                "tables": [],
                "structured_tables": [],
                "code_blocks": [],
                "api_endpoints": [],
                "view_mappings": [],
                "list_items": []
            }
            
            # Process headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_level = int(heading.name[1])
                heading_text = heading.text.strip()
                if heading_text:
                    extracted["text_blocks"].append({
                        "type": "heading",
                        "level": heading_level,
                        "text": heading_text
                    })
            
            # Process paragraphs
            for p in soup.find_all('p'):
                p_text = p.text.strip()
                if p_text:
                    extracted["text_blocks"].append({
                        "type": "paragraph",
                        "text": p_text
                    })
            
            # Process lists
            for list_tag in soup.find_all(['ul', 'ol']):
                list_type = 'unordered' if list_tag.name == 'ul' else 'ordered'
                for item in list_tag.find_all('li'):
                    item_text = item.text.strip()
                    if item_text:
                        extracted["list_items"].append({
                            "type": list_type,
                            "text": item_text
                        })
            
            # Process tables - critical for mapping data
            for i, table in enumerate(soup.find_all('table')):
                # Get table title/caption or use index
                table_title = table.find('caption')
                table_title = table_title.text.strip() if table_title else f"Table {i+1}"
                
                # Determine table purpose from title or headers
                table_purpose = "generic"
                if re.search(r'map|view\s+to\s+api|column', table_title.lower()):
                    table_purpose = "mapping"
                elif re.search(r'api|endpoint|rest', table_title.lower()):
                    table_purpose = "api"
                
                # Extract headers
                headers = []
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
                
                # If no thead, try first row
                if not headers:
                    first_row = table.find('tr')
                    if first_row:
                        first_cells = first_row.find_all(['th', 'td'])
                        if first_row.find('th') or any('bold' in cell.get('style', '') for cell in first_cells):
                            headers = [cell.text.strip() for cell in first_cells]
                
                # Process rows
                rows = []
                skip_first = bool(headers and not thead)
                table_rows = table.find_all('tr')
                
                for idx, tr in enumerate(table_rows):
                    if skip_first and idx == 0:
                        continue
                    
                    row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
                
                # Create structured table representation
                structured_table = {
                    "title": table_title,
                    "purpose": table_purpose,
                    "headers": headers,
                    "rows": rows
                }
                extracted["structured_tables"].append(structured_table)
                
                # Format as text for context
                table_text = ContentExtractor.format_table_as_text(structured_table)
                extracted["tables"].append(table_text)
                
                # Process mapping tables
                if table_purpose == "mapping" or any(re.search(r'view|api|column|attribute', h.lower()) for h in headers):
                    mapping_data = ContentExtractor.extract_mapping_from_table(structured_table)
                    if mapping_data:
                        extracted["view_mappings"].append(mapping_data)
            
            # Extract code blocks
            for pre in soup.find_all('pre'):
                code_text = pre.text.strip()
                if code_text:
                    code_type = "sql" if re.search(r'select|create|view|from|join', code_text, re.IGNORECASE) else "generic"
                    extracted["code_blocks"].append({
                        "type": code_type,
                        "code": code_text
                    })
            
            # Extract API endpoints
            api_pattern = r'(/v\d+/[a-zA-Z0-9_/{}.-]+)'
            
            for block in extracted["text_blocks"]:
                if "text" in block:
                    endpoint_matches = re.findall(api_pattern, block["text"])
                    for endpoint in endpoint_matches:
                        if not any(e["endpoint"] == endpoint for e in extracted["api_endpoints"]):
                            extracted["api_endpoints"].append({
                                "endpoint": endpoint,
                                "context": block["text"][:200]
                            })
            
            # Check structured tables for endpoints
            for table in extracted["structured_tables"]:
                # Check headers and all cells
                for header in table["headers"]:
                    endpoint_matches = re.findall(api_pattern, header)
                    for endpoint in endpoint_matches:
                        if not any(e["endpoint"] == endpoint for e in extracted["api_endpoints"]):
                            extracted["api_endpoints"].append({
                                "endpoint": endpoint,
                                "context": f"Found in table: {table['title']}"
                            })
                
                for row in table["rows"]:
                    for cell in row:
                        endpoint_matches = re.findall(api_pattern, cell)
                        for endpoint in endpoint_matches:
                            if not any(e["endpoint"] == endpoint for e in extracted["api_endpoints"]):
                                extracted["api_endpoints"].append({
                                    "endpoint": endpoint,
                                    "context": f"Found in table: {table['title']}"
                                })
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return {
                "title": title,
                "text_blocks": [{"type": "error", "text": f"Error extracting content: {str(e)}"}],
                "tables": [],
                "structured_tables": [],
                "code_blocks": [],
                "api_endpoints": [],
                "view_mappings": [],
                "list_items": []
            }
    
    @staticmethod
    def format_table_as_text(table_data):
        """Format structured table data as readable text."""
        title = table_data.get("title", "Table")
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        result = f"## {title}\n\n"
        
        if headers:
            # Calculate column widths
            col_widths = [len(str(h)) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Format header
            header_row = "| " + " | ".join([str(h).ljust(col_widths[i]) for i, h in enumerate(headers)]) + " |"
            separator = "|-" + "-|-".join(["-" * col_widths[i] for i in range(len(headers))]) + "-|"
            
            result += header_row + "\n" + separator + "\n"
            
            # Format rows
            for row in rows:
                row_cells = []
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        row_cells.append(str(cell).ljust(col_widths[i]))
                    else:
                        row_cells.append(str(cell))
                result += "| " + " | ".join(row_cells) + " |\n"
        else:
            # Table without headers
            for row in rows:
                result += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
        
        return result
    
    @staticmethod
    def extract_mapping_from_table(table_data):
        """
        Extract view-to-API mapping from a table if it appears to contain mapping data.
        
        Args:
            table_data: Structured table data
            
        Returns:
            Mapping information or None if not a mapping table
        """
        headers = table_data.get("headers", [])
        if not headers:
            return None
        
        # Look for header patterns indicating mapping table
        headers_lower = [h.lower() for h in headers]
        view_col_idx = -1
        api_col_idx = -1
        notes_col_idx = -1
        
        # Identify relevant columns
        for i, header in enumerate(headers_lower):
            if re.search(r'view|column|field|db|source', header):
                view_col_idx = i
            elif re.search(r'api|endpoint|rest|attribute|target', header):
                api_col_idx = i
            elif re.search(r'note|comment|description|detail', header):
                notes_col_idx = i
        
        # If we found view and API columns, extract mapping
        if view_col_idx >= 0 and api_col_idx >= 0:
            mappings = []
            for row in table_data.get("rows", []):
                if len(row) > max(view_col_idx, api_col_idx):
                    view_attr = row[view_col_idx].strip()
                    api_attr = row[api_col_idx].strip()
                    
                    # Only include if both fields have values
                    if view_attr and api_attr:
                        mapping = {
                            "view_attribute": view_attr,
                            "api_attribute": api_attr
                        }
                        
                        # Add notes if available
                        if notes_col_idx >= 0 and len(row) > notes_col_idx:
                            mapping["notes"] = row[notes_col_idx].strip()
                        
                        mappings.append(mapping)
            
            if mappings:
                return {
                    "table_title": table_data.get("title", ""),
                    "mappings": mappings
                }
        
        return None
    
    @staticmethod
    def format_for_context(extracted_content):
        """Format extracted content into a single text for context."""
        result = []
        
        # Add title
        if extracted_content.get("title"):
            result.append(f"# {extracted_content['title']}\n")
        
        # Add text blocks
        for block in extracted_content.get("text_blocks", []):
            if block["type"] == "heading":
                result.append(f"{'#' * block['level']} {block['text']}")
            elif block["type"] == "paragraph":
                result.append(block["text"])
        
        # Add list items
        if extracted_content.get("list_items"):
            for item in extracted_content["list_items"]:
                prefix = "- " if item["type"] == "unordered" else "1. "
                result.append(f"{prefix}{item['text']}")
        
        # Add tables
        if extracted_content.get("tables"):
            result.append("\n## Tables\n")
            for table in extracted_content["tables"]:
                result.append(table)
        
        # Add code blocks
        if extracted_content.get("code_blocks"):
            result.append("\n## Code Examples\n")
            for code_block in extracted_content["code_blocks"]:
                code_type = code_block.get("type", "")
                result.append(f"```{code_type}\n{code_block['code']}\n```")
        
        # Add API endpoints
        if extracted_content.get("api_endpoints"):
            result.append("\n## API Endpoints\n")
            for endpoint in extracted_content["api_endpoints"]:
                result.append(f"- `{endpoint['endpoint']}`: {endpoint.get('context', '')}")
        
        return "\n\n".join(result)

class ConfluenceClient:
    """Client for accessing and processing Confluence content."""
    
    def __init__(self, base_url, username, api_token, spaces=None):
        """Initialize the Confluence client."""
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.timeout = 30
        self.spaces = spaces or []
        
        # Cache structures
        self.page_cache = {}
        self.content_cache = {}
        self.space_pages = {}
        self.search_cache = {}
        
        # Create a session for connection pooling
        self.session = requests.Session()
        
        logger.info(f"Initialized Confluence client for {base_url}")
    
    def test_connection(self):
        """Test connection to Confluence API."""
        try:
            response = self.session.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                logger.info("Confluence connection successful")
                return True
            
            logger.error(f"Confluence connection failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Confluence connection error: {str(e)}")
            return False
    
    def get_page_by_id(self, page_id, expand=None):
        """Get a specific page by ID."""
        cache_key = f"page_{page_id}_{expand}"
        if cache_key in self.page_cache:
            return self.page_cache[cache_key]
        
        try:
            params = {}
            if expand:
                params["expand"] = expand
            
            response = self.session.get(
                f"{self.api_url}/content/{page_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                self.page_cache[cache_key] = result
                return result
            
            logger.error(f"Failed to get page {page_id}: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting page {page_id}: {str(e)}")
            return None
    
    def get_page_content(self, page_id):
        """Get processed content of a page."""
        if page_id in self.content_cache:
            return self.content_cache[page_id]
        
        try:
            page = self.get_page_by_id(page_id, expand="body.storage,metadata.labels")
            
            if not page:
                return None
            
            # Extract basic metadata
            space_path = ""
            if "_expandable" in page and "space" in page["_expandable"]:
                space_path = page["_expandable"]["space"].split('/')[-1]
            
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "space": space_path,
                "url": f"{self.base_url}/wiki/spaces/{space_path}/pages/{page.get('id')}"
            }
            
            # Get HTML content
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process content
            extracted = ContentExtractor.extract_content(html_content, page.get("title", ""))
            formatted = ContentExtractor.format_for_context(extracted)
            
            result = {
                "metadata": metadata,
                "content": extracted,
                "formatted": formatted
            }
            
            self.content_cache[page_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing page {page_id}: {str(e)}")
            return None
    
    def load_critical_pages(self):
        """Load all critical documentation pages."""
        loaded_pages = []
        for key, page_info in IMPORTANT_PAGES.items():
            page_id = page_info["id"]
            logger.info(f"Loading critical page: {page_info['title']} ({page_id})")
            page_content = self.get_page_content(page_id)
            if page_content:
                loaded_pages.append(page_info["title"])
        
        logger.info(f"Loaded {len(loaded_pages)} critical pages: {', '.join(loaded_pages)}")
        return len(loaded_pages)
    
    def get_all_pages_in_space(self, space_key, limit=NO_LIMIT):
        """Get all pages in a Confluence space without arbitrary limits."""
        cache_key = f"space_pages_{space_key}"
        if space_key in self.space_pages:
            return self.space_pages[space_key]
        
        try:
            all_pages = []
            start = 0
            batch_size = 100  # Maximum allowed by Confluence API
            
            logger.info(f"Fetching all pages from space: {space_key}")
            
            while True:
                try:
                    response = self.session.get(
                        f"{self.api_url}/content",
                        auth=self.auth,
                        headers=self.headers,
                        params={
                            "spaceKey": space_key,
                            "expand": "version",
                            "limit": batch_size,
                            "start": start
                        },
                        timeout=self.timeout,
                        verify=False
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Failed to get pages for space {space_key}: {response.status_code}")
                        break
                    
                    data = response.json()
                    results = data.get("results", [])
                    all_pages.extend(results)
                    
                    logger.info(f"Fetched {len(results)} pages from space {space_key}, batch starting at {start}")
                    
                    # Check if we've reached the end
                    if len(results) < batch_size or len(all_pages) >= limit:
                        break
                    
                    start += batch_size
                    time.sleep(0.1)  # Slight delay to avoid rate limiting
                    
                except Exception as e:
                    logger.error(f"Error in batch fetch for space {space_key} starting at {start}: {str(e)}")
                    break
            
            logger.info(f"Total pages loaded from space {space_key}: {len(all_pages)}")
            self.space_pages[space_key] = all_pages
            return all_pages
            
        except Exception as e:
            logger.error(f"Error getting pages for space {space_key}: {str(e)}")
            self.space_pages[space_key] = []
            return []
    
    def load_all_spaces(self):
        """Load all pages from all configured spaces."""
        total_pages = 0
        for space_key in self.spaces:
            pages = self.get_all_pages_in_space(space_key)
            total_pages += len(pages)
        
        logger.info(f"Total pages loaded from all spaces: {total_pages}")
        return total_pages
    
    def search_content(self, query, max_results=20):
        """Search for content across all spaces."""
        # Create a cache key based on query
        cache_key = hashlib.md5(query.encode('utf-8')).hexdigest()
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # First ensure all spaces are loaded
        if not self.space_pages:
            self.load_all_spaces()
        
        # Normalize query for searching
        query_norm = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_norm))
        
        # Search across all spaces
        all_candidates = []
        
        for space_key, pages in self.space_pages.items():
            for page in pages:
                # Look for term matches in title
                title = page.get("title", "").lower()
                score = 0
                
                # Score title matches highly
                for term in query_terms:
                    if term in title:
                        score += 5
                    elif term in title.replace("_", " "):
                        score += 3
                
                # Special boost for critical documentation
                page_id = page.get("id")
                for _, page_info in IMPORTANT_PAGES.items():
                    if page_id == page_info["id"]:
                        score += 3
                        break
                
                # Add pages with high title relevance or any page containing key terms
                if score > 0 or re.search(r'api|view|mapping|copper|endpoint', title, re.IGNORECASE):
                    all_candidates.append((page, score, space_key))
        
        # Sort by score and take top candidates for content analysis
        candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:max_results * 2]
        
        # Get content for candidates in parallel for deeper analysis
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {executor.submit(self.get_page_content, page["id"]): (page, score, space_key) 
                             for page, score, space_key in candidates}
            
            for future in concurrent.futures.as_completed(future_to_page):
                page, score, space_key = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        # Calculate deeper content relevance
                        final_score = score
                        page_text = content["formatted"].lower()
                        
                        # Add scores for term frequency in content
                        for term in query_terms:
                            term_count = page_text.count(term)
                            final_score += min(term_count * 0.2, 10)  # Cap at 10 points per term
                        
                        # Bonus for exact phrase match
                        if len(query_terms) > 1 and query_norm in page_text:
                            final_score += 10
                        
                        # Extra boost for view mapping content
                        if "view" in query_norm and content["content"].get("view_mappings"):
                            final_score += 15
                        
                        # Extra boost for API endpoint content
                        if "api" in query_norm and content["content"].get("api_endpoints"):
                            final_score += 15
                        
                        results.append((page, content, final_score, space_key))
                except Exception as e:
                    logger.error(f"Error analyzing content for page {page['id']}: {str(e)}")
        
        # Sort by final score and limit results
        final_results = sorted(results, key=lambda x: x[2], reverse=True)[:max_results]
        
        # Cache the search results
        self.search_cache[cache_key] = final_results
        
        return final_results
    
    def find_view_mapping(self, view_name):
        """Find mapping information for a specific view."""
        if not view_name:
            return None
            
        # Normalize view name
        view_name_norm = view_name.upper().strip()
        
        # First check critical mapping page
        mapping_page_id = IMPORTANT_PAGES.get("view_to_api_mapping", {}).get("id")
        
        if mapping_page_id:
            mapping_page = self.get_page_content(mapping_page_id)
            if mapping_page:
                # Find matches in mapping tables
                for mapping in mapping_page["content"].get("view_mappings", []):
                    # Check each mapping entry for this view
                    view_matches = []
                    for entry in mapping.get("mappings", []):
                        view_attr = entry.get("view_attribute", "").upper()
                        if view_name_norm in view_attr:
                            view_matches.append(entry)
                    
                    if view_matches:
                        return {
                            "source": "mapping_page",
                            "view_name": view_name,
                            "mappings": view_matches,
                            "page_id": mapping_page_id
                        }
        
        # Search more broadly
        search_results = self.search_content(f"{view_name} mapping", max_results=8)
        
        for page, content, score, space_key in search_results:
            for mapping in content["content"].get("view_mappings", []):
                view_matches = []
                for entry in mapping.get("mappings", []):
                    view_attr = entry.get("view_attribute", "").upper()
                    if view_name_norm in view_attr:
                        view_matches.append(entry)
                
                if view_matches:
                    return {
                        "source": f"search_page_{page['id']}",
                        "view_name": view_name,
                        "mappings": view_matches,
                        "page_id": page['id'],
                        "page_title": page['title']
                    }
        
        # If exact view mapping not found, collect similar views for reference
        similar_search = self.search_content("view mapping", max_results=10)
        
        all_mappings = []
        for page, content, score, space_key in similar_search:
            if content["content"].get("view_mappings"):
                all_mappings.append({
                    "mappings": content["content"]["view_mappings"],
                    "page_id": page['id'],
                    "page_title": page['title']
                })
        
        if all_mappings:
            return {
                "source": "similar_mappings",
                "view_name": view_name,
                "similar_mappings": all_mappings
            }
        
        return None

class StashClient:
    """Client for interacting with Bitbucket (Stash) repositories."""
    
    def __init__(self, base_url, api_token, project_key, repo_slug, directory_path):
        """Initialize the Stash client."""
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.project_key = project_key
        self.repo_slug = repo_slug
        self.directory_path = directory_path
        self.headers = {"Accept": "application/json"}
        
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Caches
        self.file_cache = {}
        self.file_list_cache = None
        
        # Mock SQL store for fallback/demonstration
        self.mock_sql_store = {
            "W_CORE_TCC_SPAN_MAPPING": """
            CREATE OR REPLACE VIEW W_CORE_TCC_SPAN_MAPPING AS 
            SELECT 
                SPAN.TCC_ID,
                SPAN.PRODUCT_ID,
                SPAN.EFFECTIVE_DATE,
                SPAN.SPAN_AMT,
                SPAN.SPAN_CURRENCY,
                SPAN.SPAN_TYPE,
                SPAN.SPAN_MODEL,
                PRODUCT.PRODUCT_CODE,
                PRODUCT.INSTRUMENT_ID,
                PRODUCT.EXCHANGE_ID,
                PRODUCT.PRODUCT_TYPE,
                CURRENCY.CURRENCY_CODE
            FROM TRD_CORE_TCC_SPAN SPAN
            JOIN PRODUCT_MASTER PRODUCT ON SPAN.PRODUCT_ID = PRODUCT.PRODUCT_ID
            JOIN CURRENCY_MASTER CURRENCY ON SPAN.SPAN_CURRENCY = CURRENCY.CURRENCY_ID
            WHERE SPAN.ACTIVE_FLAG = 'Y'
            """,
            
            "STATIC_VW_CHEDIRECT_INSTRUMENT": """
            CREATE OR REPLACE VIEW STATIC_VW_CHEDIRECT_INSTRUMENT AS
            SELECT 
                INST.INSTRUMENT_ID,
                INST.INSTRUMENT_CODE,
                INST.INSTRUMENT_TYPE,
                INST.PRODUCT_ID,
                INST.EXCHANGE_ID,
                INST.PRICE_MULT_FACTOR,
                INST.CONTRACT_SIZE,
                INST.TICK_SIZE,
                PROD.PRODUCT_CODE,
                PROD.PRODUCT_TYPE,
                EXCH.EXCHANGE_CODE,
                EXCH.EXCHANGE_NAME
            FROM INSTRUMENT_MASTER INST
            JOIN PRODUCT_MASTER PROD ON INST.PRODUCT_ID = PROD.PRODUCT_ID
            JOIN EXCHANGE_MASTER EXCH ON INST.EXCHANGE_ID = EXCH.EXCHANGE_ID
            WHERE INST.STATUS = 'ACTIVE'
            """,
            
            "W_TRD_TRADE": """
            CREATE OR REPLACE VIEW W_TRD_TRADE AS
            SELECT 
                TRADE.TRADE_ID,
                TRADE.TRADE_NO,
                TRADE.TRADE_DATE,
                TRADE.TRADE_TIME,
                TRADE.QUANTITY,
                TRADE.PRICE,
                TRADE.TRADE_TYPE,
                TRADE.TRADE_SOURCE,
                TRADE.STATUS,
                TRADE.LAST_UPDATED,
                FIRM.FIRM_ID,
                FIRM.FIRM_CODE,
                USER.USER_ID,
                USER.USERNAME,
                PRODUCT.PRODUCT_ID,
                PRODUCT.PRODUCT_CODE,
                INSTRUMENT.INSTRUMENT_ID,
                INSTRUMENT.INSTRUMENT_CODE
            FROM TRD_TRADE TRADE
            JOIN TRD_FIRM FIRM ON TRADE.FIRM_ID = FIRM.FIRM_ID
            JOIN TRD_USER USER ON TRADE.USER_ID = USER.USER_ID
            JOIN PRODUCT_MASTER PRODUCT ON TRADE.PRODUCT_ID = PRODUCT.PRODUCT_ID
            JOIN INSTRUMENT_MASTER INSTRUMENT ON TRADE.INSTRUMENT_ID = INSTRUMENT.INSTRUMENT_ID
            """
        }
        
        logger.info(f"Initialized Stash client for {base_url}")
    
    def test_connection(self):
        """Test connection to Stash API."""
        try:
            url = f"{self.base_url}/rest/api/1.0/projects/{self.project_key}"
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                logger.info("Stash connection successful")
                return True
            
            logger.warning(f"Stash connection warning: {response.status_code}")
            return False
            
        except Exception as e:
            logger.warning(f"Stash connection error: {str(e)}")
            return False
    
    def get_file_content(self, filename):
        """Get content of a specific file."""
        if not filename:
            return None
        
        # Check cache first
        if filename in self.file_cache:
            return self.file_cache[filename]
        
        # Normalize filename
        if not filename.endswith('.sql'):
            filename = f"{filename}.sql"
        
        try:
            # Form the file path
            file_path = f"{self.directory_path}/{filename}"
            url = f"{self.base_url}/rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}/raw/{file_path}"
            
            logger.info(f"Fetching file: {file_path}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                content = response.text
                self.file_cache[filename] = content
                return content
            
            logger.warning(f"Failed to get file {filename}: {response.status_code}")
            
            # Check mock store if real API failed
            base_name = filename.split('/')[-1].split('.')[0].upper()
            if base_name in self.mock_sql_store:
                logger.info(f"Using mock SQL for {base_name}")
                mock_sql = self.mock_sql_store[base_name]
                self.file_cache[filename] = mock_sql
                return mock_sql
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file {filename}: {str(e)}")
            
            # Use mock data as fallback
            base_name = filename.split('/')[-1].split('.')[0].upper()
            if base_name in self.mock_sql_store:
                logger.info(f"Using mock SQL for {base_name} after error")
                mock_sql = self.mock_sql_store[base_name]
                self.file_cache[filename] = mock_sql
                return mock_sql
                
            return None
    
    def list_files(self, force_refresh=False):
        """List all SQL files in the directory."""
        if self.file_list_cache is not None and not force_refresh:
            return self.file_list_cache
        
        try:
            url = f"{self.base_url}/rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}/files/{self.directory_path}"
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                files = response.json()
                sql_files = [f for f in files if f.endswith('.sql')]
                self.file_list_cache = sql_files
                logger.info(f"Found {len(sql_files)} SQL files")
                return sql_files
            
            # If API fails, use mock data
            logger.warning(f"Failed to list files: {response.status_code}")
            mock_files = [f"{name}.sql" for name in self.mock_sql_store.keys()]
            self.file_list_cache = mock_files
            return mock_files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            # Return mock files as fallback
            mock_files = [f"{name}.sql" for name in self.mock_sql_store.keys()]
            self.file_list_cache = mock_files
            return mock_files
    
    def get_view_sql(self, view_name):
        """Get SQL definition for a view by name."""
        if not view_name:
            return None
            
        # Normalize view name
        view_name = view_name.upper().strip()
        
        # Check mock store first
        if view_name in self.mock_sql_store:
            logger.info(f"Found view {view_name} in mock store")
            return self.mock_sql_store[view_name]
        
        # Try exact filename match
        sql = self.get_file_content(view_name)
        if sql:
            return sql
        
        # Try without .sql extension
        if view_name.endswith('.SQL'):
            sql = self.get_file_content(view_name[:-4])
            if sql:
                return sql
        
        # List all files and look for matches
        files = self.list_files()
        for filename in files:
            base_name = filename.split('/')[-1].split('.')[0].upper()
            if base_name == view_name:
                return self.get_file_content(filename)
            
            # Also check similar names (without underscores)
            if base_name.replace("_", "") == view_name.replace("_", ""):
                return self.get_file_content(filename)
        
        # If nothing found, check if any file contains this view name
        for filename in files:
            content = self.get_file_content(filename)
            if content and re.search(rf'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+{view_name}', 
                                    content, re.IGNORECASE):
                return content
        
        logger.warning(f"SQL not found for view: {view_name} - generating generic SQL")
        
        # Generate generic SQL if nothing found
        generic_sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT 
            t1.ID,
            t1.NAME,
            t1.CODE,
            t1.DESCRIPTION,
            t1.STATUS,
            t1.TYPE,
            t1.CREATED_DATE,
            t1.UPDATED_DATE,
            t2.CATEGORY,
            t2.VALUE,
            t2.CURRENCY
        FROM PRIMARY_TABLE t1
        JOIN SECONDARY_TABLE t2 ON t1.ID = t2.PRIMARY_ID
        WHERE t1.STATUS = 'ACTIVE'
        """
        
        return generic_sql

class SQLParser:
    """Parse and analyze SQL view definitions."""
    
    @staticmethod
    def parse_view(sql_text):
        """Parse SQL view definition into structured components."""
        if not sql_text:
            return None
        
        try:
            # Clean up SQL text
            sql_text = sql_text.strip()
            
            # Extract view name
            view_name_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([^\s(]+)', 
                                       sql_text, re.IGNORECASE)
            view_name = view_name_match.group(1) if view_name_match else "UnknownView"
            
            # Extract columns from SELECT clause
            columns = []
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_text, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                
                # Handle SELECT *
                if '*' in select_clause:
                    columns.append({
                        "name": "*",
                        "alias": None,
                        "table": None
                    })
                else:
                    # Split by comma but respect parentheses
                    column_expressions = []
                    current_expr = ""
                    paren_level = 0
                    
                    for char in select_clause:
                        if char == '(':
                            paren_level += 1
                        elif char == ')':
                            paren_level -= 1
                        
                        if char == ',' and paren_level == 0:
                            column_expressions.append(current_expr.strip())
                            current_expr = ""
                        else:
                            current_expr += char
                    
                    if current_expr.strip():
                        column_expressions.append(current_expr.strip())
                    
                    # Process each column
                    for expr in column_expressions:
                        # Check for AS keyword
                        as_match = re.search(r'(.*?)\s+AS\s+([A-Za-z0-9_]+)\s*$', expr, re.IGNORECASE)
                        if as_match:
                            col_expr = as_match.group(1).strip()
                            alias = as_match.group(2).strip()
                            
                            # Check if table.column format
                            if '.' in col_expr and not '(' in col_expr:
                                parts = col_expr.split('.')
                                table = parts[0].strip()
                                name = parts[1].strip()
                                columns.append({
                                    "name": name,
                                    "alias": alias,
                                    "table": table,
                                    "expression": expr
                                })
                            else:
                                columns.append({
                                    "name": col_expr,
                                    "alias": alias,
                                    "table": None,
                                    "expression": expr
                                })
                        else:
                            # No AS keyword - check for implicit alias
                            if re.search(r'[A-Za-z0-9_.]+\s+[A-Za-z0-9_]+\s*$', expr) and not '(' in expr:
                                parts = expr.split()
                                col_expr = ' '.join(parts[:-1])
                                alias = parts[-1]
                                
                                if '.' in col_expr:
                                    table_col = col_expr.split('.')
                                    table = table_col[0].strip()
                                    name = table_col[1].strip()
                                    columns.append({
                                        "name": name,
                                        "alias": alias,
                                        "table": table,
                                        "expression": expr
                                    })
                                else:
                                    columns.append({
                                        "name": col_expr,
                                        "alias": alias,
                                        "table": None,
                                        "expression": expr
                                    })
                            else:
                                # Simple column reference
                                if '.' in expr:
                                    parts = expr.split('.')
                                    table = parts[0].strip()
                                    name = parts[1].strip()
                                    columns.append({
                                        "name": name,
                                        "alias": None,
                                        "table": table,
                                        "expression": expr
                                    })
                                else:
                                    columns.append({
                                        "name": expr,
                                        "alias": None,
                                        "table": None,
                                        "expression": expr
                                    })
            
            # Extract tables from FROM/JOIN clauses
            tables = []
            joins = []
            from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|$)', 
                                  sql_text, re.IGNORECASE | re.DOTALL)
            
            if from_match:
                from_clause = from_match.group(1)
                
                # Extract main table first
                main_table_match = re.search(r'^\s*([A-Za-z0-9_."]+)(?:\s+(?:AS\s+)?([A-Za-z0-9_]+))?', 
                                           from_clause, re.IGNORECASE)
                if main_table_match:
                    table_name = main_table_match.group(1).strip('"')
                    alias = main_table_match.group(2) if main_table_match.group(2) else None
                    
                    tables.append({
                        "name": table_name,
                        "alias": alias,
                        "join_type": "FROM"
                    })
                
                # Extract joins
                join_pattern = r'(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+([A-Za-z0-9_."]+)(?:\s+(?:AS\s+)?([A-Za-z0-9_]+))?\s+ON\s+(.+?)(?=\s+(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN|\s+WHERE|\s*$)'
                join_matches = re.finditer(join_pattern, from_clause, re.IGNORECASE | re.DOTALL)
                
                for match in join_matches:
                    join_type = match.group(1) if match.group(1) else "INNER"
                    table_name = match.group(2).strip('"')
                    alias = match.group(3) if match.group(3) else None
                    condition = match.group(4).strip()
                    
                    tables.append({
                        "name": table_name,
                        "alias": alias,
                        "join_type": f"{join_type} JOIN"
                    })
                    
                    joins.append({
                        "type": join_type,
                        "table": table_name,
                        "alias": alias,
                        "condition": condition
                    })
            
            # Extract WHERE clauses
            where_clauses = []
            where_match = re.search(r'WHERE\s+(.*?)(?:GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|$)', 
                                   sql_text, re.IGNORECASE | re.DOTALL)
            
            if where_match:
                where_clause = where_match.group(1)
                # Split on AND (simplified approach)
                conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
                for condition in conditions:
                    condition = condition.strip()
                    if condition:
                        where_clauses.append(condition)
            
            return {
                "view_name": view_name,
                "columns": columns,
                "tables": tables,
                "joins": joins,
                "where_clauses": where_clauses,
                "sql_text": sql_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {str(e)}")
            return {
                "view_name": view_name if 'view_name' in locals() else "ErrorView",
                "columns": [],
                "tables": [],
                "joins": [],
                "where_clauses": [],
                "sql_text": sql_text,
                "error": str(e)
            }
    
    @staticmethod
    def categorize_columns(parsed_view):
        """Categorize columns by purpose and data type."""
        categorized = []
        
        if not parsed_view or "columns" not in parsed_view:
            return categorized
        
        for column in parsed_view["columns"]:
            category = "unknown"
            name = column.get("name", "").lower()
            alias = column.get("alias", "").lower() if column.get("alias") else name
            display_name = alias or name
            data_type = "string"
            
            # Determine category based on naming patterns
            if re.search(r'id$|guid$|key$', display_name):
                category = "identifier"
                data_type = "string"
            elif re.search(r'date$|time$|timestamp$', display_name):
                category = "datetime"
                data_type = "timestamp"
            elif re.search(r'amt$|amount$|sum$|total$|price$|qty$|quantity$', display_name):
                category = "numeric"
                data_type = "number"
            elif re.search(r'name$|desc|description$|label$|title$', display_name):
                category = "descriptive"
                data_type = "string"
            elif re.search(r'flag$|indicator$|status$|type$|category$|code$', display_name):
                category = "code"
                data_type = "string"
            elif re.search(r'currency$|curr$', display_name):
                category = "currency"
                data_type = "string"
            
            categorized.append({
                "column": column,
                "category": category,
                "data_type": data_type
            })
        
        return categorized

class MappingGenerator:
    """Generate API mappings for database views."""
    
    def __init__(self, confluence_client, stash_client, gemini_model):
        """Initialize the mapping generator."""
        self.confluence = confluence_client
        self.stash = stash_client
        self.model = gemini_model
        self.sql_parser = SQLParser()
        self.mapping_cache = {}
        
        # Common mapping patterns
        self.mapping_patterns = {
            # Field suffix → API field patterns
            "ID": "id",
            "GUID": "guid",
            "CODE": "code",
            "NAME": "name",
            "DESC": "description",
            "AMOUNT": "amount",
            "QTY": "quantity",
            "DATE": "date",
            "TIME": "time",
            "TIMESTAMP": "timestamp",
            "FLAG": "indicator",
            "STATUS": "status",
            "TYPE": "type"
        }
        
        # API endpoint patterns
        self.endpoint_patterns = {
            "product": "/v1/products",
            "instrument": "/v1/instruments",
            "trade": "/v1/trades",
            "session": "/v1/sessions",
            "firm": "/v1/firms",
            "user": "/v1/users",
            "exchange": "/v1/exchanges",
            "tcc": "/v1/tcc",  # For TCC_SPAN_MAPPING
            "ticket": "/v1/tickets"
        }
        
        logger.info("Initialized Mapping Generator")
    
    def generate_mapping(self, view_name=None, sql_text=None):
        """Generate complete mapping for a view."""
        # Either view_name or sql_text must be provided
        if not view_name and not sql_text:
            logger.error("Either view_name or sql_text must be provided")
            return None
        
        # If only view name is provided, get SQL from Stash
        if not sql_text and view_name:
            sql_text = self.stash.get_view_sql(view_name)
            if not sql_text:
                logger.error(f"Could not get SQL for view: {view_name}")
                return None
        
        # Generate a cache key based on input
        cache_key = hashlib.md5((view_name or "") + (sql_text or "")).hexdigest()
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]
        
        # Parse the SQL
        parsed_view = self.sql_parser.parse_view(sql_text)
        if not parsed_view:
            logger.error("Failed to parse SQL")
            return None
        
        # Use view name from SQL if not provided
        if not view_name:
            view_name = parsed_view["view_name"]
        else:
            # Ensure view name consistency
            parsed_view["view_name"] = view_name
        
        # Multi-strategy approach to generate mapping:
        # 1. Try to find mapping in Confluence documentation
        # 2. If incomplete, augment with pattern-based mapping
        # 3. If still insufficient, use AI-based mapping
        
        # Strategy 1: Documentation-based mapping
        doc_mapping = self.confluence.find_view_mapping(view_name)
        
        if doc_mapping and doc_mapping.get("source") != "similar_mappings" and doc_mapping.get("mappings"):
            logger.info(f"Found mapping in documentation for {view_name}")
            mapping_result = self.format_doc_mapping(doc_mapping, parsed_view)
            
            # Cache and return
            self.mapping_cache[cache_key] = mapping_result
            return mapping_result
        
        # Strategy 2: Pattern-based mapping
        logger.info(f"Generating pattern-based mapping for {view_name}")
        pattern_mapping = self.generate_pattern_mapping(parsed_view)
        
        # Strategy 3: If pattern mapping is insufficient, use AI-based mapping
        if len(pattern_mapping.get("attribute_mappings", [])) < len(parsed_view.get("columns", [])) * 0.6:
            logger.info(f"Pattern mapping insufficient, using AI for {view_name}")
            ai_mapping = self.generate_ai_mapping(parsed_view, doc_mapping)
            mapping_result = ai_mapping
        else:
            mapping_result = pattern_mapping
        
        # Cache and return
        self.mapping_cache[cache_key] = mapping_result
        return mapping_result
    
    def format_doc_mapping(self, doc_mapping, parsed_view):
        """Format documentation-based mapping into standard structure."""
        result = {
            "view_name": doc_mapping.get("view_name", parsed_view["view_name"]),
            "api_endpoints": [],
            "attribute_mappings": [],
            "request_body": {},
            "response_body": {},
            "source": "documentation"
        }
        
        # Extract API endpoints from mappings
        endpoints = set()
        for mapping in doc_mapping.get("mappings", []):
            api_attr = mapping.get("api_attribute", "")
            # Look for endpoint patterns
            endpoint_match = re.search(r'(/v\d+/[a-zA-Z0-9_/{}.-]+)', api_attr)
            if endpoint_match:
                endpoints.add(endpoint_match.group(1))
        
        # Add all unique endpoints
        for endpoint in endpoints:
            result["api_endpoints"].append({
                "endpoint": endpoint,
                "description": "Found in documentation mapping"
            })
        
        # If no endpoints found, try to infer from view name
        if not result["api_endpoints"]:
            inferred_endpoint = self.infer_endpoint_from_view(parsed_view)
            if inferred_endpoint:
                result["api_endpoints"].append({
                    "endpoint": inferred_endpoint,
                    "description": "Inferred from view name and structure"
                })
        
        # Process attribute mappings
        for mapping in doc_mapping.get("mappings", []):
            view_attr = mapping.get("view_attribute", "")
            api_attr = mapping.get("api_attribute", "")
            
            # Normalize API attribute - remove endpoint prefix if present
            api_attr_clean = re.sub(r'^/v\d+/[a-zA-Z0-9_/{}.-]+\s*[:]\s*', '', api_attr)
            
            # Add to attribute mappings
            result["attribute_mappings"].append({
                "view_attribute": view_attr,
                "api_attribute": api_attr_clean,
                "notes": mapping.get("notes", "")
            })
        
        # Generate sample request and response bodies
        result["request_body"] = self.generate_request_body(result["attribute_mappings"])
        result["response_body"] = self.generate_response_body(result["attribute_mappings"])
        
        return result
    
    def generate_pattern_mapping(self, parsed_view):
        """Generate mapping based on naming patterns and conventions."""
        result = {
            "view_name": parsed_view["view_name"],
            "api_endpoints": [],
            "attribute_mappings": [],
            "request_body": {},
            "response_body": {},
            "source": "pattern_matching"
        }
        
        # Infer API endpoint from view name and structure
        inferred_endpoint = self.infer_endpoint_from_view(parsed_view)
        if inferred_endpoint:
            result["api_endpoints"].append({
                "endpoint": inferred_endpoint,
                "description": "Inferred from view name and structure"
            })
        
        # Categorize columns for better mapping
        categorized_columns = SQLParser.categorize_columns(parsed_view)
        
        # Generate attribute mappings based on common patterns
        for cat_col in categorized_columns:
            column = cat_col["column"]
            category = cat_col["category"]
            
            column_name = column.get("name", "")
            alias = column.get("alias", "") or column_name
            display_name = alias or column_name
            
            # Convert to camelCase for API attribute
            api_attr = self.to_camel_case(display_name)
            
            # Special handling based on category
            if category == "identifier":
                if display_name.upper().endswith("_ID"):
                    base_name = re.sub(r'_ID$', '', display_name, flags=re.IGNORECASE)
                    api_attr = self.to_camel_case(f"{base_name}Id")
            elif category == "datetime":
                if display_name.upper().endswith("_DATE"):
                    base_name = re.sub(r'_DATE$', '', display_name, flags=re.IGNORECASE)
                    api_attr = self.to_camel_case(f"{base_name}Date")
            
            # Check for column name patterns
            for suffix, api_pattern in self.mapping_patterns.items():
                if display_name.upper().endswith(suffix):
                    base_name = re.sub(f"{suffix}$", "", display_name, flags=re.IGNORECASE)
                    if base_name:
                        api_attr = self.to_camel_case(f"{base_name}_{api_pattern}")
                    else:
                        api_attr = api_pattern
                    break
            
            # Add mapping
            result["attribute_mappings"].append({
                "view_attribute": display_name,
                "api_attribute": api_attr,
                "notes": f"Category: {category}, Data type: {cat_col.get('data_type', 'string')}"
            })
        
        # Generate sample request and response bodies
        result["request_body"] = self.generate_request_body(result["attribute_mappings"])
        result["response_body"] = self.generate_response_body(result["attribute_mappings"])
        
        return result
    
    def generate_ai_mapping(self, parsed_view, doc_mapping=None):
        """Generate mapping using AI model."""
        logger.info(f"Generating AI-based mapping for {parsed_view['view_name']}")
        
        # Get relevant API documentation
        api_context = self.get_api_context()
        
        # Prepare SQL analysis for the model
        sql_analysis = {
            "view_name": parsed_view["view_name"],
            "columns": parsed_view["columns"],
            "tables": parsed_view["tables"],
            "joins": parsed_view["joins"],
            "where_clauses": parsed_view["where_clauses"]
        }
        
        # Add any documentation mapping that might help
        doc_context = ""
        if doc_mapping:
            if doc_mapping.get("source") == "similar_mappings":
                doc_context = "Here are mappings for similar views that might help:\n"
                for i, mapping in enumerate(doc_mapping.get("similar_mappings", [])):
                    doc_context += f"Similar mapping {i+1} from {mapping.get('page_title', 'Unknown')}:\n"
                    for m in mapping.get("mappings", []):
                        for item in m.get("mappings", []):
                            doc_context += f"- View: {item.get('view_attribute', '')} → API: {item.get('api_attribute', '')}\n"
                    doc_context += "\n"
            else:
                doc_context = "Partial documentation mapping found:\n"
                for item in doc_mapping.get("mappings", []):
                    doc_context += f"- View: {item.get('view_attribute', '')} → API: {item.get('api_attribute', '')}\n"
                doc_context += "\n"
        
        # Prepare prompt for the model
        prompt = f"""
        You are an expert at mapping database views to REST API endpoints and attributes for the COPPER API system.
        
        I need to map this SQL view to the appropriate COPPER API endpoints and attributes:
        
        ```sql
        {parsed_view['sql_text']}
        ```
        
        SQL View Analysis:
        {json.dumps(sql_analysis, indent=2)}
        
        {doc_context}
        
        COPPER API documentation:
        {api_context}
        
        Please create a comprehensive mapping that:
        1. Identifies the most appropriate API endpoint(s) for this view
        2. Maps each view column to the corresponding API attribute
        3. Provides a sample API request body
        4. Provides a sample API response body
        
        Rules for mapping:
        - Convert database column names to camelCase for API attributes
        - Tables typically map to API endpoints (e.g., PRODUCT_MASTER → /v1/products)
        - Database fields with _ID suffix typically become camelCase with Id suffix
        - Views that combine multiple tables may map to the primary entity's endpoint
        - Nested objects can be used to represent related entities
        
        Return your response as a detailed JSON object with this structure:
        {{
            "view_name": "Name of the view",
            "api_endpoints": [
                {{ "endpoint": "/v1/example", "description": "Description of the endpoint" }}
            ],
            "attribute_mappings": [
                {{ "view_attribute": "COLUMN_NAME", "api_attribute": "columnName", "notes": "Additional information" }}
            ],
            "request_body": {{ 
                // Sample request body with camelCase attributes
            }},
            "response_body": {{
                // Sample response body with all mapped attributes
            }}
        }}
        """
        
        try:
            # Generate mapping with AI
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8000
                )
            )
            
            if response.candidates and response.candidates[0].text:
                result_text = response.candidates[0].text.strip()
                
                # Extract JSON from response
                try:
                    # Find JSON object in response
                    json_pattern = r'```json\s*([\s\S]*?)\s*```'
                    json_match = re.search(json_pattern, result_text)
                    
                    if json_match:
                        json_text = json_match.group(1)
                    else:
                        # Try to find JSON object directly
                        json_pattern = r'({[\s\S]*})'
                        json_match = re.search(json_pattern, result_text)
                        if json_match:
                            json_text = json_match.group(1)
                        else:
                            json_text = result_text
                    
                    # Parse JSON
                    result = json.loads(json_text)
                    result["source"] = "ai_generated"
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing AI response as JSON: {str(e)}")
                    # Fall back to pattern-based mapping
                    return self.generate_pattern_mapping(parsed_view)
                
            else:
                logger.error("No response from AI model")
                return self.generate_pattern_mapping(parsed_view)
                
        except Exception as e:
            logger.error(f"Error generating AI mapping: {str(e)}")
            return self.generate_pattern_mapping(parsed_view)
    
    def get_api_context(self):
        """Get relevant API documentation context."""
        context = []
        
        # Load critical API documentation pages
        for key, page_info in IMPORTANT_PAGES.items():
            if "api" in key.lower() or "mapping" in key.lower():
                page_content = self.confluence.get_page_content(page_info["id"])
                if page_content:
                    # Extract relevant sections
                    content_text = page_content["formatted"]
                    # Limit size for each page
                    if len(content_text) > 2000:
                        content_text = content_text[:2000] + "..."
                    context.append(f"## {page_info['title']}\n{content_text}")
        
        return "\n\n".join(context)
    
    def infer_endpoint_from_view(self, parsed_view):
        """Infer API endpoint from view name and structure."""
        view_name = parsed_view.get("view_name", "").upper()
        
        # Extract domain from view name
        domain = None
        
        # Look for common domain patterns in view name
        for key, _ in self.endpoint_patterns.items():
            if key.upper() in view_name:
                domain = key
                break
        
        # If not found in name, look at tables
        if not domain:
            # Count domain references in table names
            domain_counts = {}
            for pattern in self.endpoint_patterns.keys():
                domain_counts[pattern] = 0
            
            for table in parsed_view.get("tables", []):
                table_name = table.get("name", "").upper()
                for pattern in self.endpoint_patterns.keys():
                    if pattern.upper() in table_name:
                        domain_counts[pattern] += 1
            
            # Find most common domain
            max_count = 0
            for pattern, count in domain_counts.items():
                if count > max_count:
                    max_count = count
                    domain = pattern
        
        # If we found a domain, return its endpoint
        if domain and domain in self.endpoint_patterns:
            return self.endpoint_patterns[domain]
        
        # Default: Extract from view name
        clean_name = re.sub(r'^(W_|STATIC_VW_)', '', view_name)
        parts = clean_name.split('_')
        
        # Look for meaningful resource name
        resource = None
        for part in parts:
            if part.lower() not in ['core', 'static', 'vw']:
                resource = part.lower()
                break
        
        if resource:
            return f"/v1/{resource}s"
        
        # Fallback
        return f"/v1/{clean_name.lower()}"
    
    def to_camel_case(self, snake_case):
        """Convert snake_case or UPPER_CASE to camelCase."""
        if not snake_case:
            return ""
            
        # Handle special cases
        if snake_case.upper() == snake_case:
            # All uppercase, convert to lowercase first
            snake_case = snake_case.lower()
        
        # Replace special characters with underscores
        clean_str = re.sub(r'[^a-zA-Z0-9_]', '_', snake_case)
        
        # Convert to camelCase
        components = clean_str.split('_')
        return components[0].lower() + ''.join(x.title() for x in components[1:] if x)
    
    def generate_request_body(self, attribute_mappings):
        """Generate a sample request body based on attribute mappings."""
        body = {}
        
        # Process attribute mappings
        for mapping in attribute_mappings:
            api_attr = mapping.get("api_attribute", "")
            view_attr = mapping.get("view_attribute", "")
            
            # Skip if empty or invalid
            if not api_attr or re.search(r'[^a-zA-Z0-9_.]', api_attr):
                continue
            
            # Generate sample value based on attribute name
            value = self.generate_sample_value(api_attr, view_attr)
            
            # Handle nested attributes (with dots)
            if "." in api_attr:
                parts = api_attr.split(".")
                current = body
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        current[part] = value
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            else:
                body[api_attr] = value
        
        return body
    
    def generate_response_body(self, attribute_mappings):
        """Generate a sample response body with additional standard fields."""
        response = self.generate_request_body(attribute_mappings)
        
        # Add standard API response fields if not already present
        if isinstance(response, dict):
            if not any(k in ["id", "guid"] for k in response.keys()):
                response["id"] = "12345"
            
            if not any(k in ["createdAt", "createdDate"] for k in response.keys()):
                response["createdAt"] = "2024-04-04T12:00:00Z"
                
            if not any(k in ["updatedAt", "lastUpdated"] for k in response.keys()):
                response["updatedAt"] = "2024-04-04T12:30:00Z"
        
        return response
    
    def generate_sample_value(self, api_attr, view_attr):
        """Generate appropriate sample value based on attribute name."""
        attr_lower = api_attr.lower()
        
        # Use naming patterns to determine appropriate sample values
        if re.search(r'id$|guid$', attr_lower):
            return "12345"
        elif re.search(r'date$|time$|timestamp$', attr_lower):
            return "2024-04-04T12:00:00Z"
        elif re.search(r'amount$|price$|quantity$|size$|factor$', attr_lower):
            return 100.00
        elif re.search(r'name$', attr_lower):
            return "Sample Name"
        elif re.search(r'description$|desc$', attr_lower):
            return "Sample description"
        elif re.search(r'code$|type$|status$', attr_lower):
            return "ACTIVE"
        elif re.search(r'flag$|indicator$', attr_lower):
            return True
        elif re.search(r'currency$', attr_lower):
            return "USD"
        elif re.search(r'exchange$', attr_lower):
            return "CME"
        else:
            return "sample_value"

class CopperAssistant:
    """Main class for the COPPER View to API Mapper."""
    
    def __init__(self):
        """Initialize the COPPER Assistant."""
        self.initialized = False
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=PROJECT_ID, location=REGION)
            self.model = GenerativeModel(MODEL_NAME)
            
            # Initialize clients
            self.confluence = ConfluenceClient(
                CONFLUENCE_URL, 
                CONFLUENCE_USERNAME, 
                CONFLUENCE_API_TOKEN, 
                spaces=CONFLUENCE_SPACES
            )
            
            self.stash = StashClient(
                BITBUCKET_URL,
                BITBUCKET_API_TOKEN,
                PROJECT_KEY,
                REPO_SLUG,
                DIRECTORY_PATH
            )
            
            # Initialize mapping generator
            self.mapping_generator = MappingGenerator(
                self.confluence,
                self.stash,
                self.model
            )
            
            # Conversation memory for context
            self.memory = []
            
            # Response cache
            self.response_cache = {}
            
            logger.info("COPPER Assistant initialized successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing COPPER Assistant: {str(e)}")
            print(f"Initialization error: {str(e)}")
    
    def preload_data(self):
        """Preload critical data for faster responses."""
        try:
            logger.info("Preloading critical data...")
            
            # Load important Confluence pages
            pages_loaded = self.confluence.load_critical_pages()
            logger.info(f"Loaded {pages_loaded} critical Confluence pages")
            
            # Start loading all spaces in background
            threading.Thread(target=self.confluence.load_all_spaces, daemon=True).start()
            
            # Test Stash connection and preload file list
            stash_connected = self.stash.test_connection()
            logger.info(f"Stash connection status: {stash_connected}")
            
            # Preload view list
            files = self.stash.list_files()
            logger.info(f"Preloaded {len(files)} SQL files from repository")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preloading data: {str(e)}")
            return False
    
    def answer_question(self, question):
        """
        Answer a user question about COPPER views and APIs.
        
        Args:
            question: The user's question
            
        Returns:
            The answer
        """
        if not self.initialized:
            return "Sorry, I'm not properly initialized. Please check the logs for errors."
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Create a cache key for this question
            cache_key = hashlib.md5(question.encode('utf-8')).hexdigest()
            if cache_key in self.response_cache:
                logger.info("Returning cached response")
                return self.response_cache[cache_key]
            
            # Detect intent to determine how to process
            intent = self.detect_intent(question)
            logger.info(f"Detected intent: {intent}")
            
            if intent == "view_mapping":
                # Extract view name
                view_name = self.extract_view_name(question)
                if view_name:
                    logger.info(f"Generating mapping for view: {view_name}")
                    response = self.generate_view_mapping_response(view_name)
                else:
                    response = "I couldn't identify which view you're asking about. Please specify the view name clearly."
            
            elif intent == "sql_mapping":
                # Extract SQL query
                sql = self.extract_sql(question)
                if sql:
                    logger.info("Generating mapping from SQL")
                    response = self.generate_sql_mapping_response(sql)
                else:
                    response = "I couldn't find a SQL query in your question. Please provide the view definition."
            
            elif intent == "api_usage":
                logger.info("Generating API usage information")
                response = self.generate_api_usage_response(question)
            
            else:
                # General question
                logger.info("Generating general response")
                response = self.generate_general_response(question)
            
            # Cache the response
            self.response_cache[cache_key] = response
            
            # Update conversation memory
            self.update_memory(question, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def detect_intent(self, question):
        """Detect the intent of the user's question."""
        question_lower = question.lower()
        
        # Check for view mapping intent
        view_patterns = [
            r"mapping\s+for\s+(?:view|table)?",
            r"map\s+(?:view|table)",
            r"view\s+to\s+api",
            r"which\s+api\s+(?:for|to\s+use\s+with)\s+(?:view|table)",
            r"what\s+is\s+the\s+mapping\s+for"
        ]
        for pattern in view_patterns:
            if re.search(pattern, question_lower):
                return "view_mapping"
        
        # Check for SQL mapping intent
        sql_patterns = [
            r"create\s+(?:or\s+replace\s+)?view",
            r"select\s+.*?\s+from",
            r"map\s+this\s+sql",
            r"generate\s+mapping\s+for\s+sql"
        ]
        for pattern in sql_patterns:
            if re.search(pattern, question_lower):
                return "sql_mapping"
        
        # Check for API usage intent
        api_patterns = [
            r"how\s+(?:do|to|can)\s+I\s+use",
            r"which\s+api",
            r"what\s+is\s+the\s+endpoint",
            r"how\s+to\s+call",
            r"api\s+for\s+(?!view)",
            r"supported\s+operators"
        ]
        for pattern in api_patterns:
            if re.search(pattern, question_lower):
                return "api_usage"
        
        # Default to general question
        return "general"
    
    def extract_view_name(self, question):
        """Extract view name from question."""
        # Try common patterns
        patterns = [
            r"(?:view|table)\s+(?:named|called)\s+([A-Za-z0-9_]+)",
            r"(?:for|of|to)\s+(?:view|table)?\s+([A-Za-z0-9_]+)",
            r"mapping\s+(?:for|of)\s+([A-Za-z0-9_]+)",
            r"([A-Za-z0-9_]+)\s+view\s+to\s+api",
            r"W_[A-Za-z0-9_]+"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for words that match common view name patterns
        words = re.findall(r'\b[A-Za-z0-9_]+\b', question)
        for word in words:
            if word.startswith('W_') or word.upper() == word:
                return word
        
        return None
    
    def extract_sql(self, question):
        """Extract SQL query from user input."""
        # Look for code blocks
        code_block_patterns = [
            r"```sql\s*([\s\S]*?)\s*```",
            r"```\s*(CREATE[\s\S]*?VIEW[\s\S]*?)\s*```",
            r"```\s*(SELECT[\s\S]*?FROM[\s\S]*?)\s*```"
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for inline SQL
        sql_patterns = [
            r"(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+\w+\s+AS[\s\S]*?)(?:;|$|Please)",
            r"(SELECT[\s\S]*?FROM[\s\S]*?)(?:;|$|Please)"
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def generate_view_mapping_response(self, view_name):
        """Generate mapping response for a specific view."""
        # Get mapping
        mapping = self.mapping_generator.generate_mapping(view_name=view_name)
        
        if not mapping:
            return f"I couldn't generate a mapping for view {view_name}. Please check if the view name is correct."
        
        # Format the response
        response = f"# Mapping for {view_name}\n\n"
        
        # API Endpoints
        if mapping.get("api_endpoints"):
            response += "## API Endpoints\n\n"
            for endpoint in mapping.get("api_endpoints"):
                response += f"- `{endpoint.get('endpoint')}`"
                if endpoint.get("description"):
                    response += f": {endpoint.get('description')}"
                response += "\n"
            response += "\n"
        
        # Attribute Mappings
        if mapping.get("attribute_mappings"):
            response += "## Attribute Mappings\n\n"
            response += "| View Attribute | API Attribute | Notes |\n"
            response += "|---------------|--------------|-------|\n"
            
            for attr in mapping.get("attribute_mappings"):
                view_attr = attr.get("view_attribute", "")
                api_attr = attr.get("api_attribute", "")
                notes = attr.get("notes", "")
                response += f"| {view_attr} | {api_attr} | {notes} |\n"
            
            response += "\n"
        
        # Sample Request Body
        if mapping.get("request_body"):
            response += "## Sample Request Body\n\n"
            response += "```json\n"
            response += json.dumps(mapping.get("request_body"), indent=2)
            response += "\n```\n\n"
        
        # Sample Response Body
        if mapping.get("response_body"):
            response += "## Sample Response Body\n\n"
            response += "```json\n"
            response += json.dumps(mapping.get("response_body"), indent=2)
            response += "\n```\n\n"
        
        # Source information
        source = mapping.get("source", "unknown")
        if source == "documentation":
            response += "*This mapping was found in the COPPER documentation.*\n"
        elif source == "pattern_matching":
            response += "*This mapping was generated based on naming patterns. Please verify before use.*\n"
        elif source == "ai_generated":
            response += "*This mapping was generated using AI based on the view definition and API documentation. Please verify before use.*\n"
        
        return response
    
    def generate_sql_mapping_response(self, sql):
        """Generate mapping response from SQL definition."""
        # Parse the SQL to extract view name if possible
        parsed = SQLParser.parse_view(sql)
        view_name = parsed.get("view_name") if parsed else "UnknownView"
        
        # Generate mapping
        mapping = self.mapping_generator.generate_mapping(sql_text=sql)
        
        if not mapping:
            return "I couldn't generate a mapping for this SQL. Please check if the SQL is valid."
        
        # Format the response
        response = f"# Mapping for SQL View: {view_name}\n\n"
        
        # API Endpoints
        if mapping.get("api_endpoints"):
            response += "## API Endpoints\n\n"
            for endpoint in mapping.get("api_endpoints"):
                response += f"- `{endpoint.get('endpoint')}`"
                if endpoint.get("description"):
                    response += f": {endpoint.get('description')}"
                response += "\n"
            response += "\n"
        
        # Attribute Mappings
        if mapping.get("attribute_mappings"):
            response += "## Attribute Mappings\n\n"
            response += "| View Attribute | API Attribute | Notes |\n"
            response += "|---------------|--------------|-------|\n"
            
            for attr in mapping.get("attribute_mappings"):
                view_attr = attr.get("view_attribute", "")
                api_attr = attr.get("api_attribute", "")
                notes = attr.get("notes", "")
                response += f"| {view_attr} | {api_attr} | {notes} |\n"
            
            response += "\n"
        
        # Sample Request Body
        if mapping.get("request_body"):
            response += "## Sample Request Body\n\n"
            response += "```json\n"
            response += json.dumps(mapping.get("request_body"), indent=2)
            response += "\n```\n\n"
        
        # Sample Response Body
        if mapping.get("response_body"):
            response += "## Sample Response Body\n\n"
            response += "```json\n"
            response += json.dumps(mapping.get("response_body"), indent=2)
            response += "\n```\n\n"
        
        # Source information
        source = mapping.get("source", "unknown")
        if source == "documentation":
            response += "*This mapping was found in the COPPER documentation.*\n"
        elif source == "pattern_matching":
            response += "*This mapping was generated based on naming patterns. Please verify before use.*\n"
        elif source == "ai_generated":
            response += "*This mapping was generated using AI based on the SQL definition and API documentation. Please verify before use.*\n"
        
        return response
    
    def generate_api_usage_response(self, question):
        """Generate response about API usage."""
        # Get relevant API documentation
        search_results = self.confluence.search_content(question, max_results=5)
        
        context = []
        for page, content, score, space in search_results:
            context.append(content["formatted"])
        
        # Additional context from specific pages
        api_topics = ["endpoint", "api", "operator"]
        for topic in api_topics:
            if topic in question.lower():
                for key, page_info in IMPORTANT_PAGES.items():
                    if topic in key.lower():
                        page_content = self.confluence.get_page_content(page_info["id"])
                        if page_content:
                            context.append(page_content["formatted"])
        
        # Add conversation context
        conversation_context = self.get_conversation_context()
        
        # Generate response with AI
        prompt = f"""
        You are a COPPER API expert working at CME. A colleague has asked you this question:
        
        {question}
        
        Here is relevant information from the COPPER API documentation:
        
        {' '.join(context[:3])}
        
        Previous conversation:
        {conversation_context}
        
        Give a clear, helpful response in a professional but conversational tone. Include specific API endpoints, parameters, and examples where relevant. Format your response using Markdown. Be concise but thorough, and focus directly on answering the question.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )
            
            if response.candidates and response.candidates[0].text:
                return response.candidates[0].text.strip()
            else:
                return "I couldn't find specific information about that in the COPPER API documentation. Could you try rephrasing your question or providing more details?"
                
        except Exception as e:
            logger.error(f"Error generating API usage response: {str(e)}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def generate_general_response(self, question):
        """Generate response for general questions."""
        # Get relevant documentation
        search_results = self.confluence.search_content(question, max_results=8)
        
        context = []
        for page, content, score, space in search_results[:5]:  # Use top 5 results
            context.append(f"## {page['title']}\n{content['formatted'][:2000]}")
        
        # Add conversation context
        conversation_context = self.get_conversation_context()
        
        # Generate response with AI
        prompt = f"""
        You are a COPPER expert working at CME. A colleague has asked you this question:
        
        {question}
        
        Here is relevant information from the COPPER documentation:
        
        {' '.join(context)}
        
        Previous conversation:
        {conversation_context}
        
        Give a clear, helpful response in a professional but conversational tone. Format your response using Markdown. Be concise but thorough, and focus directly on answering the question.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )
            
            if response.candidates and response.candidates[0].text:
                return response.candidates[0].text.strip()
            else:
                return "I couldn't find specific information about that in the COPPER documentation. Could you try rephrasing your question or providing more details?"
                
        except Exception as e:
            logger.error(f"Error generating general response: {str(e)}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def update_memory(self, question, response):
        """Update conversation memory with latest exchange."""
        self.memory.append({
            "question": question,
            "response": response,
            "timestamp": time.time()
        })
        
        # Keep only last 5 exchanges
        if len(self.memory) > 5:
            self.memory.pop(0)
    
    def get_conversation_context(self):
        """Get formatted conversation history for context."""
        if not self.memory:
            return "No previous conversation."
        
        context = []
        for i, exchange in enumerate(self.memory[-3:]):  # Last 3 exchanges
            context.append(f"Question {i+1}: {exchange['question']}")
            # Limit response length
            response = exchange['response']
            if len(response) > 200:
                response = response[:200] + "..."
            context.append(f"Answer {i+1}: {response}")
        
        return "\n\n".join(context)

def main():
    """Main function to run the COPPER View to API Mapper."""
    print("\n=== COPPER View to API Mapper ===\n")
    print("Initializing...")
    
    # Initialize the assistant
    assistant = CopperAssistant()
    
    if not assistant.is_initialized():
        print("Error: Failed to initialize. Please check the logs for details.")
        return
    
    # Preload critical data
    print("Loading documentation and SQL views...")
    assistant.preload_data()
    
    print("\n=== COPPER View to API Mapper ===")
    print("I can help you map COPPER database views to REST APIs.")
    print("\nExamples of what you can ask:")
    print("  - What is the mapping for view W_CORE_TCC_SPAN_MAPPING?")
    print("  - Which API can I use to get product data?")
    print("  - How do I find all instruments for an exchange?")
    print("  - What operators are supported for timestamps?")
    print("  - Map this SQL: CREATE VIEW my_view AS SELECT...")
    print("\nType 'exit' to quit.")
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using the COPPER View to API Mapper. Goodbye!")
                break
                
            if not query:
                continue
                
            print("\nWorking on your request...")
            start_time = time.time()
            
            response = assistant.answer_question(query)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"\nResponse (generated in {elapsed:.2f} seconds):")
            print("=" * 80)
            print(response)
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\nOperation interrupted. Exiting...")
            break
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
