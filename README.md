#!/usr/bin/env python3
# Enhanced COPPER View to API Mapper

"""
Advanced COPPER View to API Mapper
-----------------------------------
Integrates with Confluence and Bitbucket to:
1. Extract structured data from Confluence documentation
2. Retrieve SQL view definitions from Bitbucket
3. Parse and analyze SQL queries
4. Generate comprehensive view-to-API mappings
5. Create API request body examples
6. Answer COPPER API questions using Gemini AI
"""

import os
import sys
import json
import re
import time
import logging
import hashlib
import concurrent.futures
import threading
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import sqlparse
from urllib.parse import urljoin

# GenAI imports
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
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
MAX_WORKERS = 5
CACHE_SIZE = 1000

# Critical COPPER API Documentation Pages - direct access with page IDs
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
        Extract structured content from HTML with advanced parsing for tables.
        
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
                "view_mappings": []
            }
            
            # Process headings to maintain document structure
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
            
            # Process tables with enhanced metadata detection
            for i, table in enumerate(soup.find_all('table')):
                # Get table title/caption
                table_title = table.find('caption')
                table_title = table_title.text.strip() if table_title else f"Table {i+1}"
                
                # Determine table purpose from title or nearby headings
                table_purpose = "unknown"
                if re.search(r'map|endpoint|api|view', table_title, re.IGNORECASE):
                    table_purpose = "mapping"
                
                # Extract headers
                headers = []
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
                else:
                    # Try first row if no thead
                    first_row = table.find('tr')
                    if first_row:
                        headers = [cell.text.strip() for cell in first_row.find_all(['th', 'td'])]
                        # Check if these actually look like headers
                        if not any(re.search(r'api|view|endpoint|field|column|attribute', 
                                            header, re.IGNORECASE) for header in headers):
                            headers = []
                
                # Process rows
                rows = []
                tbody = table.find('tbody')
                data_rows = tbody.find_all('tr') if tbody else table.find_all('tr')
                
                # Skip header row if we extracted headers and it's the first row
                start_idx = 1 if headers and data_rows and not thead else 0
                
                for tr in data_rows[start_idx:]:
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
                
                # Also create formatted text version for context
                text_table = ContentExtractor.format_table_as_text(structured_table)
                extracted["tables"].append(text_table)
                
                # If this looks like a mapping table, process it specially
                if table_purpose == "mapping" or any(re.search(r'api|endpoint|view', h, re.IGNORECASE) for h in headers):
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
            for text_block in extracted["text_blocks"]:
                if "text" in text_block:
                    endpoint_matches = re.findall(api_pattern, text_block["text"])
                    for endpoint in endpoint_matches:
                        if endpoint not in [e["endpoint"] for e in extracted["api_endpoints"]]:
                            # Get context from surrounding text
                            context = text_block["text"][:200]  # Just use the first part of the text
                            extracted["api_endpoints"].append({
                                "endpoint": endpoint,
                                "context": context
                            })
            
            # Also check tables for API endpoints
            for table in extracted["structured_tables"]:
                for row in table["rows"]:
                    for cell in row:
                        endpoint_matches = re.findall(api_pattern, cell)
                        for endpoint in endpoint_matches:
                            if endpoint not in [e["endpoint"] for e in extracted["api_endpoints"]]:
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
                "view_mappings": []
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
            if re.search(r'view|column|field|db', header):
                view_col_idx = i
            elif re.search(r'api|endpoint|rest|attribute', header):
                api_col_idx = i
            elif re.search(r'note|comment|description', header):
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
        self.cache = {}
        self.content_cache = {}
        self.spaces = spaces or []
        self.space_pages = {}
        
        logger.info(f"Initialized Confluence client for {base_url}")
    
    def test_connection(self):
        """Test connection to Confluence API."""
        try:
            response = requests.get(
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
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            params = {}
            if expand:
                params["expand"] = expand
            
            response = requests.get(
                f"{self.api_url}/content/{page_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                self.cache[cache_key] = result
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
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "space": page.get("_expandable", {}).get("space", "").split("/")[-1],
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}"
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
        for key, page_info in IMPORTANT_PAGES.items():
            page_id = page_info["id"]
            logger.info(f"Loading critical page: {page_info['title']} ({page_id})")
            self.get_page_content(page_id)
    
    def get_all_pages_in_space(self, space_key, limit=500):
        """Get all pages in a Confluence space."""
        cache_key = f"space_pages_{space_key}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            all_pages = []
            start = 0
            while True:
                response = requests.get(
                    f"{self.api_url}/content",
                    auth=self.auth,
                    headers=self.headers,
                    params={
                        "spaceKey": space_key,
                        "expand": "version",
                        "limit": 100,
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
                
                if len(results) < 100 or len(all_pages) >= limit:
                    break
                
                start += 100
                time.sleep(0.5)  # Avoid rate limiting
            
            logger.info(f"Loaded {len(all_pages)} pages from space {space_key}")
            self.cache[cache_key] = all_pages
            return all_pages
            
        except Exception as e:
            logger.error(f"Error getting pages for space {space_key}: {str(e)}")
            return []
    
    def load_all_spaces(self):
        """Load all pages from configured spaces."""
        for space_key in self.spaces:
            self.space_pages[space_key] = self.get_all_pages_in_space(space_key)
            logger.info(f"Loaded {len(self.space_pages[space_key])} pages from space {space_key}")
    
    def search_content(self, query, max_results=10):
        """Search for content across spaces."""
        # First check if we need to load spaces
        if not self.space_pages:
            self.load_all_spaces()
        
        all_candidates = []
        
        # Normalize query for searching
        query_norm = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_norm))
        
        # Search in each space
        for space_key, pages in self.space_pages.items():
            for page in pages:
                title = page.get("title", "").lower()
                
                # Calculate title match score
                title_score = 0
                for term in query_terms:
                    if term in title:
                        title_score += 3
                
                # Add to candidates if good title match
                if title_score > 0 or re.search(r'api|view|mapping|copper', title, re.IGNORECASE):
                    all_candidates.append((page, title_score, space_key))
        
        # Sort by score
        candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:20]
        
        # Get content for candidates
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {executor.submit(self.get_page_content, page["id"]): (page, score, space_key) 
                             for page, score, space_key in candidates}
            
            for future in concurrent.futures.as_completed(future_to_page):
                page, score, space_key = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        # Calculate content relevance
                        relevance = score
                        page_text = content["formatted"].lower()
                        
                        # Add term frequency score
                        for term in query_terms:
                            term_count = page_text.count(term)
                            relevance += min(term_count * 0.1, 3)  # Cap at 3 points per term
                        
                        # Bonus for exact phrase match
                        if len(query_terms) > 1 and query_norm in page_text:
                            relevance += 5
                        
                        results.append((page, content, relevance, space_key))
                except Exception as e:
                    logger.error(f"Error getting content for page {page['id']}: {str(e)}")
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
    
    def find_view_mapping(self, view_name):
        """Find mapping information for a specific view."""
        # First check critical mapping page
        mapping_page_id = IMPORTANT_PAGES.get("view_to_api_mapping", {}).get("id")
        
        if mapping_page_id:
            mapping_page = self.get_page_content(mapping_page_id)
            if mapping_page:
                # Find matches in tables
                for mapping in mapping_page["content"].get("view_mappings", []):
                    # Check each mapping entry for this view
                    view_matches = []
                    for entry in mapping.get("mappings", []):
                        view_attr = entry.get("view_attribute", "").lower()
                        if view_name.lower() in view_attr:
                            view_matches.append(entry)
                    
                    if view_matches:
                        return {
                            "source": "mapping_page",
                            "view_name": view_name,
                            "mappings": view_matches
                        }
        
        # If not found in mapping page, search more broadly
        search_results = self.search_content(f"{view_name} mapping")
        
        for page, content, score, space_key in search_results:
            for mapping in content["content"].get("view_mappings", []):
                view_matches = []
                for entry in mapping.get("mappings", []):
                    view_attr = entry.get("view_attribute", "").lower()
                    if view_name.lower() in view_attr:
                        view_matches.append(entry)
                
                if view_matches:
                    return {
                        "source": f"search_page_{page['id']}",
                        "view_name": view_name,
                        "mappings": view_matches
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
        
        self.session = requests.Session()
        self.cache = {}
        
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
            
            logger.error(f"Stash connection failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Stash connection error: {str(e)}")
            return False
    
    def get_file_content(self, filename):
        """Get content of a specific file."""
        if not filename:
            return None
        
        cache_key = f"file_{filename}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Form the file path
            if not filename.endswith('.sql'):
                filename = f"{filename}.sql"
            
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
                self.cache[cache_key] = content
                return content
            
            logger.error(f"Failed to get file {filename}: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting file {filename}: {str(e)}")
            return None
    
    def list_files(self):
        """List all SQL files in the directory."""
        cache_key = "file_list"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
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
                self.cache[cache_key] = sql_files
                logger.info(f"Found {len(sql_files)} SQL files")
                return sql_files
            
            # If the API fails, try an alternative approach - mock files for testing
            logger.warning(f"Failed to list files: {response.status_code} - Using mock list")
            mock_files = [
                "W_CORE_TCC_SPAN_MAPPING.sql",
                "STATIC_VW_CHEDIRECT_INSTRUMENT.sql",
                "W_TRD_TRADE.sql"
            ]
            self.cache[cache_key] = mock_files
            return mock_files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            # Return mock files for demonstration
            mock_files = [
                "W_CORE_TCC_SPAN_MAPPING.sql",
                "STATIC_VW_CHEDIRECT_INSTRUMENT.sql",
                "W_TRD_TRADE.sql"
            ]
            self.cache[cache_key] = mock_files
            return mock_files
    
    def get_view_sql(self, view_name):
        """Get SQL definition for a view by name."""
        # Normalize view name
        view_name = view_name.upper().strip()
        
        # Try exact filename match first
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
        
        # If nothing found, check if any file contains this view name
        for filename in files:
            content = self.get_file_content(filename)
            if content and re.search(rf'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+{view_name}', 
                                    content, re.IGNORECASE):
                return content
        
        logger.warning(f"SQL not found for view: {view_name}")
        
        # If still not found, provide a mock SQL for testing
        logger.info(f"Generating mock SQL for view: {view_name}")
        return f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT 
            t1.field1 AS FIELD_ONE,
            t1.field2 AS FIELD_TWO,
            t2.field3 AS FIELD_THREE
        FROM table1 t1
        JOIN table2 t2 ON t1.id = t2.id
        WHERE t1.active = 'Y'
        """

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
            
            # Parse SQL with sqlparse
            parsed = sqlparse.parse(sql_text)
            if not parsed:
                return {
                    "view_name": view_name,
                    "columns": [],
                    "tables": [],
                    "joins": [],
                    "where_clauses": [],
                    "sql_text": sql_text
                }
                
            statement = parsed[0]
            
            # Extract columns from SELECT clause
            columns = []
            select_found = False
            from_found = False
            column_tokens = []
            
            for token in statement.tokens:
                if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
                    select_found = True
                    continue
                
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    from_found = True
                    break
                
                if select_found and not from_found and token.ttype not in sqlparse.tokens.Whitespace:
                    if token.ttype is sqlparse.tokens.Punctuation and token.value == ',':
                        pass  # Skip commas
                    else:
                        column_tokens.append(token)
            
            # Process column tokens
            for token in column_tokens:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    # Multiple columns
                    for identifier in token.get_identifiers():
                        col = SQLParser.process_column_identifier(identifier)
                        if col:
                            columns.append(col)
                elif isinstance(token, sqlparse.sql.Identifier):
                    # Single column
                    col = SQLParser.process_column_identifier(token)
                    if col:
                        columns.append(col)
            
            # Extract tables and join conditions
            tables = []
            joins = []
            where_clauses = []
            
            # Look for FROM clause
            from_clause = ""
            where_clause = ""
            for token in statement.tokens:
                if isinstance(token, sqlparse.sql.Where):
                    where_clause = token.value
                    break
                    
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    # Find the FROM clause
                    idx = statement.tokens.index(token)
                    for t in statement.tokens[idx+1:]:
                        if t.ttype is sqlparse.tokens.Keyword and t.value.upper() in ['WHERE', 'GROUP', 'ORDER', 'HAVING']:
                            break
                        if t.ttype not in sqlparse.tokens.Whitespace:
                            from_clause += t.value
            
            # Extract tables from FROM clause (simplified approach)
            table_pattern = r'(?:FROM|JOIN)\s+([A-Za-z0-9_."]+)(?:\s+(?:AS\s+)?([A-Za-z0-9_]+))?'
            table_matches = re.finditer(table_pattern, sql_text, re.IGNORECASE)
            
            for match in table_matches:
                table_name = match.group(1).strip('"')
                alias = match.group(2) if match.group(2) else None
                
                tables.append({
                    "name": table_name,
                    "alias": alias
                })
            
            # Extract JOIN conditions
            join_pattern = r'(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+([A-Za-z0-9_."]+)(?:\s+(?:AS\s+)?([A-Za-z0-9_]+))?\s+ON\s+(.+?)(?=\s+(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN|\s+WHERE|\s*$)'
            join_matches = re.finditer(join_pattern, sql_text, re.IGNORECASE | re.DOTALL)
            
            for match in join_matches:
                join_type = match.group(1) if match.group(1) else "INNER"
                table_name = match.group(2).strip('"')
                alias = match.group(3) if match.group(3) else None
                condition = match.group(4).strip()
                
                joins.append({
                    "type": join_type,
                    "table": table_name,
                    "alias": alias,
                    "condition": condition
                })
            
            # Extract WHERE conditions
            if where_clause:
                # Split on AND/OR (simplified approach)
                conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
                for condition in conditions:
                    condition = condition.strip()
                    if condition and not condition.upper().startswith('WHERE'):
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
                "view_name": "ErrorView",
                "columns": [],
                "tables": [],
                "joins": [],
                "where_clauses": [],
                "sql_text": sql_text,
                "error": str(e)
            }
    
    @staticmethod
    def process_column_identifier(identifier):
        """Process a column identifier token and extract column info."""
        if not identifier:
            return None
        
        try:
            column_name = identifier.value
            
            # Handle AS clauses
            if " AS " in column_name:
                parts = column_name.split(" AS ")
                column_name = parts[0].strip()
                alias = parts[1].strip()
            else:
                alias = None
            
            # Handle table.column format
            if "." in column_name:
                parts = column_name.split(".")
                table = parts[0].strip()
                column = parts[1].strip()
            else:
                table = None
                column = column_name.strip()
            
            return {
                "name": column,
                "alias": alias,
                "table": table,
                "full_text": identifier.value
            }
            
        except Exception as e:
            logger.error(f"Error processing column identifier: {str(e)}")
            return {
                "name": str(identifier),
                "alias": None,
                "table": None,
                "error": str(e)
            }
    
    @staticmethod
    def categorize_view_columns(parsed_view):
        """Categorize columns by purpose based on naming patterns."""
        categorized = []
        
        if not parsed_view or "columns" not in parsed_view:
            return categorized
        
        for column in parsed_view["columns"]:
            category = "unknown"
            name = column.get("name", "").lower()
            alias = column.get("alias", "").lower() if column.get("alias") else name
            
            # Categorize based on common naming patterns
            if re.search(r'id$|guid$|key$', alias):
                category = "identifier"
            elif re.search(r'date$|time$|timestamp$', alias):
                category = "datetime"
            elif re.search(r'amt$|amount$|sum$|total$|price$|qty$', alias):
                category = "numeric"
            elif re.search(r'name$|desc|description$|label$|title$', alias):
                category = "descriptive"
            elif re.search(r'flag$|indicator$|status$|type$|category$', alias):
                category = "code"
            
            categorized.append({
                "column": column,
                "category": category
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
        
        # Load common mapping patterns
        self.mapping_patterns = {
            # Column suffix to API field patterns
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
        
        # Try to find mapping in Confluence documentation
        doc_mapping = self.confluence.find_view_mapping(view_name)
        
        # If found in documentation, use it
        if doc_mapping and doc_mapping.get("mappings"):
            logger.info(f"Found mapping in documentation for {view_name}")
            mapping_result = self.format_doc_mapping(doc_mapping, parsed_view)
            self.mapping_cache[cache_key] = mapping_result
            return mapping_result
        
        # If not found in documentation, try pattern-based mapping
        logger.info(f"Generating pattern-based mapping for {view_name}")
        pattern_mapping = self.generate_pattern_mapping(parsed_view)
        
        # If pattern mapping has few results, use AI-based mapping
        if len(pattern_mapping.get("attribute_mappings", [])) < len(parsed_view.get("columns", [])) / 2:
            logger.info(f"Pattern mapping insufficient, using AI for {view_name}")
            ai_mapping = self.generate_ai_mapping(parsed_view)
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
                "description": "From documentation mapping"
            })
        
        # If no endpoints found, try to infer from view name
        if not result["api_endpoints"]:
            inferred_endpoint = self.infer_endpoint_from_view(parsed_view["view_name"])
            if inferred_endpoint:
                result["api_endpoints"].append({
                    "endpoint": inferred_endpoint,
                    "description": "Inferred from view name"
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
        
        # Generate sample request body
        result["request_body"] = self.generate_request_body(result["attribute_mappings"])
        
        return result
    
    def generate_pattern_mapping(self, parsed_view):
        """Generate mapping based on naming patterns and conventions."""
        result = {
            "view_name": parsed_view["view_name"],
            "api_endpoints": [],
            "attribute_mappings": [],
            "request_body": {},
            "source": "pattern_matching"
        }
        
        # Infer API endpoint from view name
        inferred_endpoint = self.infer_endpoint_from_view(parsed_view["view_name"])
        if inferred_endpoint:
            result["api_endpoints"].append({
                "endpoint": inferred_endpoint,
                "description": "Inferred from view name"
            })
        
        # Generate attribute mappings based on common patterns
        for column in parsed_view.get("columns", []):
            column_name = column.get("name", "")
            alias = column.get("alias", "") or column_name
            
            # Convert to camelCase for API attribute
            api_attr = self.to_camel_case(alias)
            
            # Check for common mapping patterns
            for suffix, api_pattern in self.mapping_patterns.items():
                if column_name.upper().endswith(suffix) or (alias and alias.upper().endswith(suffix)):
                    base_name = re.sub(f"{suffix}$", "", alias, flags=re.IGNORECASE)
                    api_attr = self.to_camel_case(f"{base_name}_{api_pattern}")
                    break
            
            # Add mapping
            result["attribute_mappings"].append({
                "view_attribute": alias or column_name,
                "api_attribute": api_attr,
                "notes": f"Mapped from column {column.get('full_text', column_name)}"
            })
        
        # Generate sample request body
        result["request_body"] = self.generate_request_body(result["attribute_mappings"])
        
        return result
    
    def generate_ai_mapping(self, parsed_view):
        """Generate mapping using AI model."""
        logger.info(f"Generating AI-based mapping for {parsed_view['view_name']}")
        
        # Get relevant API documentation
        api_context = self.get_api_context()
        
        # Prepare prompt for the model
        prompt = f"""
        You are an expert at mapping database views to REST API endpoints and attributes.
        
        I need to map this SQL view to the appropriate COPPER API endpoints and attributes:
        
        ```sql
        {parsed_view['sql_text']}
        ```
        
        COPPER API documentation:
        {api_context}
        
        Please create a comprehensive mapping that:
        1. Identifies the most appropriate API endpoint(s) for this view
        2. Maps each view column to the corresponding API attribute
        3. Provides a sample API request body
        
        Return your response as a JSON object with this structure:
        {{
            "view_name": "Name of the view",
            "api_endpoints": [
                {{ "endpoint": "/v1/example", "description": "Description of the endpoint" }}
            ],
            "attribute_mappings": [
                {{ "view_attribute": "COLUMN_NAME", "api_attribute": "columnName", "notes": "Additional information" }}
            ],
            "request_body": {{ 
                "sample JSON request object with camelCase attributes" 
            }}
        }}
        """
        
        try:
            # Generate mapping with AI
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4000
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
            if "api" in key.lower():
                page_content = self.confluence.get_page_content(page_info["id"])
                if page_content:
                    # Extract relevant sections
                    content_text = page_content["formatted"]
                    # Limit size
                    if len(content_text) > 1000:
                        content_text = content_text[:1000] + "..."
                    context.append(f"## {page_info['title']}\n{content_text}")
        
        return "\n\n".join(context)
    
    def infer_endpoint_from_view(self, view_name):
        """Infer API endpoint from view name based on common patterns."""
        # Strip prefixes like W_ or STATIC_VW_
        clean_name = re.sub(r'^(W_|STATIC_VW_)', '', view_name)
        
        # Convert to lowercase with hyphens
        endpoint_base = re.sub(r'_', '-', clean_name.lower())
        
        # Common COPPER API patterns
        if "product" in endpoint_base:
            return "/v1/products"
        elif "instrument" in endpoint_base:
            return "/v1/instruments"
        elif "trade" in endpoint_base:
            return "/v1/trades"
        elif "session" in endpoint_base:
            return "/v1/sessions"
        elif "firm" in endpoint_base:
            return "/v1/firms"
        elif "user" in endpoint_base:
            return "/v1/users"
        
        # Default pattern
        return f"/v1/{endpoint_base}"
    
    def to_camel_case(self, snake_case):
        """Convert snake_case to camelCase."""
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
        
        for mapping in attribute_mappings:
            api_attr = mapping.get("api_attribute", "")
            
            # Skip if empty or contains special characters
            if not api_attr or re.search(r'[^a-zA-Z0-9_.]', api_attr):
                continue
            
            # Generate placeholder value based on attribute name
            if re.search(r'id$|guid$', api_attr.lower()):
                value = "12345"
            elif re.search(r'date$|time$', api_attr.lower()):
                value = "2024-04-04T12:00:00Z"
            elif re.search(r'amount$|price$|quantity$', api_attr.lower()):
                value = 100.00
            elif re.search(r'name$', api_attr.lower()):
                value = "Sample Name"
            elif re.search(r'description$|desc$', api_attr.lower()):
                value = "Sample description"
            elif re.search(r'code$|type$|status$', api_attr.lower()):
                value = "ACTIVE"
            elif re.search(r'flag$|indicator$', api_attr.lower()):
                value = true
            else:
                value = "value"
            
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
            
            # Conversation memory
            self.memory = []
            
            # Response cache
            self.response_cache = {}
            
            logger.info("COPPER Assistant initialized successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing COPPER Assistant: {str(e)}")
    
    def is_initialized(self):
        """Check if the assistant is properly initialized."""
        return self.initialized
    
    def preload_data(self):
        """Preload critical data."""
        try:
            logger.info("Preloading critical data...")
            
            # Load important Confluence pages
            self.confluence.load_critical_pages()
            
            # Test connection to Stash
            self.stash.test_connection()
            
            # Preload view list
            self.stash.list_files()
            
            logger.info("Preloading complete")
            return True
            
        except Exception as e:
            logger.error(f"Error preloading data: {str(e)}")
            return False
    
    def answer_question(self, query):
        """Answer a user question about COPPER views and APIs."""
        if not self.initialized:
            return "Sorry, I'm not properly initialized. Please check the logs for errors."
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Check cache first
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if cache_key in self.response_cache:
                logger.info("Returning cached response")
                return self.response_cache[cache_key]
            
            # Detect intent
            intent = self.detect_intent(query)
            logger.info(f"Detected intent: {intent}")
            
            if intent == "view_mapping":
                # Extract view name
                view_name = self.extract_view_name(query)
                if view_name:
                    logger.info(f"Generating mapping for view: {view_name}")
                    response = self.generate_view_mapping_response(view_name)
                else:
                    response = "I couldn't identify which view you're asking about. Please specify the view name clearly."
            
            elif intent == "sql_mapping":
                # Extract SQL query
                sql = self.extract_sql(query)
                if sql:
                    logger.info("Generating mapping from SQL")
                    response = self.generate_sql_mapping_response(sql)
                else:
                    response = "I couldn't find a SQL query in your question. Please provide the view definition."
            
            elif intent == "api_usage":
                logger.info("Generating API usage information")
                response = self.generate_api_usage_response(query)
            
            else:
                # General question
                logger.info("Generating general response")
                response = self.generate_general_response(query)
            
            # Cache the response
            self.response_cache[cache_key] = response
            
            # Update conversation memory
            self.memory.append({
                "query": query,
                "response": response,
                "timestamp": time.time()
            })
            
            # Keep only last 5 exchanges
            if len(self.memory) > 5:
                self.memory.pop(0)
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def detect_intent(self, query):
        """Detect the intent of the user's query."""
        query_lower = query.lower()
        
        # Check for view mapping intent
        view_patterns = [
            r"mapping\s+for\s+(?:view|table)?",
            r"map\s+(?:view|table)",
            r"view\s+to\s+api",
            r"which\s+api\s+(?:for|to\s+use\s+with)\s+(?:view|table)"
        ]
        for pattern in view_patterns:
            if re.search(pattern, query_lower):
                return "view_mapping"
        
        # Check for SQL mapping intent
        sql_patterns = [
            r"create\s+(?:or\s+replace\s+)?view",
            r"select\s+.*?\s+from",
            r"map\s+this\s+sql",
            r"generate\s+mapping\s+for\s+sql"
        ]
        for pattern in sql_patterns:
            if re.search(pattern, query_lower):
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
            if re.search(pattern, query_lower):
                return "api_usage"
        
        # Default to general question
        return "general"
    
    def extract_view_name(self, query):
        """Extract view name from query."""
        # Try common patterns
        patterns = [
            r"(?:view|table)\s+(?:named|called)\s+([A-Za-z0-9_]+)",
            r"(?:for|of|to)\s+(?:view|table)?\s+([A-Za-z0-9_]+)",
            r"mapping\s+(?:for|of)\s+([A-Za-z0-9_]+)",
            r"([A-Za-z0-9_]+)\s+view\s+to\s+api",
            r"W_[A-Za-z0-9_]+"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for words that match common view name patterns
        words = re.findall(r'\b[A-Za-z0-9_]+\b', query)
        for word in words:
            if word.startswith('W_') or word.upper() == word:
                return word
        
        return None
    
    def extract_sql(self, query):
        """Extract SQL query from user input."""
        # Look for code blocks
        code_block_patterns = [
            r"```sql\s*([\s\S]*?)\s*```",
            r"```\s*(CREATE[\s\S]*?VIEW[\s\S]*?)\s*```",
            r"```\s*(SELECT[\s\S]*?FROM[\s\S]*?)\s*```"
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for inline SQL
        sql_patterns = [
            r"(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+\w+\s+AS[\s\S]*?);",
            r"(SELECT[\s\S]*?FROM[\s\S]*?)(?:;|$)"
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
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
        
        # Source information
        source = mapping.get("source", "unknown")
        if source == "documentation":
            response += "*This mapping was found in the COPPER documentation.*\n"
        elif source == "pattern_matching":
            response += "*This mapping was generated based on naming patterns. Please verify before use.*\n"
        elif source == "ai_generated":
            response += "*This mapping was generated using AI based on the SQL definition and API documentation. Please verify before use.*\n"
        
        return response
    
    def generate_api_usage_response(self, query):
        """Generate response about API usage."""
        # Identify specific API topics
        topics = self.extract_api_topics(query)
        
        # Get relevant API documentation
        context = self.get_api_documentation(topics)
        
        # Generate response with AI
        prompt = f"""
        You are a COPPER API expert. Answer the following question about the COPPER API using the documentation provided.
        
        User question: {query}
        
        COPPER API documentation:
        {context}
        
        Provide a clear, detailed response that directly answers the user's question. Include specific API endpoints, parameters, request/response examples, and any relevant details from the documentation. Format your response using Markdown for readability.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4000
                )
            )
            
            if response.candidates and response.candidates[0].text:
                return response.candidates[0].text.strip()
            else:
                return "I couldn't find specific information about that in the COPPER API documentation."
                
        except Exception as e:
            logger.error(f"Error generating API usage response: {str(e)}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def extract_api_topics(self, query):
        """Extract API-related topics from query."""
        topics = []
        
        # Check for common API-related terms
        api_topics = {
            "product": ["product", "products"],
            "instrument": ["instrument", "instruments"],
            "trade": ["trade", "trades", "trading"],
            "session": ["session", "sessions"],
            "firm": ["firm", "firms"],
            "user": ["user", "users"],
            "timestamp": ["timestamp", "time", "date"],
            "operator": ["operator", "operators", "operations"],
            "endpoint": ["endpoint", "endpoints", "api", "rest"]
        }
        
        query_lower = query.lower()
        for topic, terms in api_topics.items():
            if any(term in query_lower for term in terms):
                topics.append(topic)
        
        return topics
    
    def get_api_documentation(self, topics=None):
        """Get relevant API documentation based on topics."""
        context = []
        
        # Priority pages
        priority_pages = ["api_faq", "api_endpoints", "api_quickstart"]
        
        # Add topic-specific pages
        if topics:
            if "operator" in topics:
                priority_pages.append("supported_operators")
            if "view" in topics or "mapping" in topics:
                priority_pages.append("view_to_api_mapping")
        
        # Load critical pages first
        for key in priority_pages:
            page_info = IMPORTANT_PAGES.get(key)
            if page_info:
                page_content = self.confluence.get_page_content(page_info["id"])
                if page_content:
                    context.append(f"## {page_info['title']}")
                    context.append(page_content["formatted"])
        
        # If specific topics, search for additional content
        if topics:
            for topic in topics:
                search_results = self.confluence.search_content(topic, max_results=2)
                for page, content, score, space_key in search_results:
                    # Skip if already included
                    if any(page["title"] in c for c in context):
                        continue
                    
                    context.append(f"## {page['title']}")
                    context.append(content["formatted"])
        
        # Limit context size
        combined = "\n\n".join(context)
        if len(combined) > 8000:
            combined = combined[:8000] + "...\n\n(Content truncated due to size)"
        
        return combined
    
    def generate_general_response(self, query):
        """Generate response for general questions."""
        # Get relevant documentation
        search_results = self.confluence.search_content(query)
        
        context = []
        for page, content, score, space_key in search_results[:3]:
            context.append(f"## {page['title']}")
            context.append(content["formatted"])
        
        # Include conversation history for context
        conversation_context = self.get_conversation_context()
        
        # Generate response with AI
        prompt = f"""
        You are a COPPER expert. Answer the following question using the documentation provided.
        
        User question: {query}
        
        COPPER documentation:
        {' '.join(context)}
        
        Previous conversation:
        {conversation_context}
        
        Provide a clear, helpful response that directly answers the user's question. If the documentation doesn't contain the specific information needed, acknowledge this and provide your best guidance based on the available information. Format your response using Markdown for readability.
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
                return "I couldn't find specific information about that in the COPPER documentation."
                
        except Exception as e:
            logger.error(f"Error generating general response: {str(e)}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def get_conversation_context(self):
        """Get formatted conversation history for context."""
        if not self.memory:
            return "No previous conversation."
        
        context = []
        for i, exchange in enumerate(self.memory[-3:]):  # Last 3 exchanges
            context.append(f"Question {i+1}: {exchange['query']}")
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
    
    # Preload data
    print("Loading documentation and API resources...")
    assistant.preload_data()
    
    print("\n=== COPPER View to API Mapper ===")
    print("I can help you map COPPER database views to REST APIs.")
    print("Examples of what you can ask:")
    print("  - What is the mapping for view W_CORE_TCC_SPAN_MAPPING?")
    print("  - Which API can I use to get product data?")
    print("  - How do I find all instruments for an exchange?")
    print("  - What operators are supported for timestamps?")
    print("  - Provide mapping for SQL: CREATE VIEW my_view AS SELECT...")
    print("\nType 'exit' to quit.")
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using the COPPER View to API Mapper. Goodbye!")
                break
                
            if not query:
                continue
                
            print("\nGenerating response...")
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
