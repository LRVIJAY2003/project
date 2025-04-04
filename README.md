#!/usr/bin/env python3
# Enhanced COPPER View to API Mapper for CME BLR Hackathon 2025

"""
Comprehensive tool that:
1. Fetches content from both Confluence spaces (COPPER documentation)
2. Retrieves SQL view definitions from Stash/Bitbucket
3. Analyzes view structure and maps to COPPER API endpoints
4. Uses Gemini AI to provide intelligent responses about mappings
5. Handles various query types about COPPER API and database views
"""

# Standard library imports
import logging
import os
import sys
import json
import re
import time
import concurrent.futures
from datetime import datetime
from functools import lru_cache
import threading
import argparse
import csv
from typing import Dict, List, Tuple, Any, Optional, Union

# Third-party imports
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sqlparse

# GenAI/Gemini AI imports
try:
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    import vertexai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Vertex AI (Gemini) modules not available. Some features will be disabled.")

# Suppress only the single warning from urllib3 needed for insecure requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("copper_mapper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CopperMapper")

# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """Manages application configuration from environment variables with defaults."""
    
    # Gemini AI / Vertex AI configuration
    PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dev-cop-4363")
    REGION = os.environ.get("REGION", "us-central1")
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-exp-83.25")
    
    # Confluence configuration
    CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://mygroup.atlassian.net")
    CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
    CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
    CONFLUENCE_SPACES = os.environ.get("CONFLUENCE_SPACES", "CE,itsrch").split(',')
    
    # Stash/Bitbucket configuration
    STASH_URL = os.environ.get("STASH_URL", "https://mystash.atlassian.net")
    STASH_TOKEN = os.environ.get("STASH_TOKEN", "")
    STASH_PROJECT_KEY = os.environ.get("STASH_PROJECT_KEY", "BLRHACKATHON") 
    STASH_REPO_SLUG = os.environ.get("STASH_REPO_SLUG", "copper-views")
    STASH_VIEW_PATH = os.environ.get("STASH_VIEW_PATH", "queries")
    
    # Cache configuration
    CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "1000"))
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "5"))
    CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
    
    # Important Confluence pages for COPPER documentation
    IMPORTANT_PAGES = {
        "introduction": "https://mygroup.atlassian.net/wiki/spaces/CE/pages/224622013/COPPER+APP/A/",
        "faq": "https://mygroup.atlassian.net/wiki/spaces/CE/cards/pages/168711190/COPPER+API+Frequently+Asked+Questions",
        "mapping": "https://mygroup.atlassian.net/wiki/spaces/CE/cards/pages/168617692/ViewtoAPIMapping",
        "quickstart": "https://mygroup.atlassian.net/wiki/spaces/CE/cards/pages/168687143/COPPER+API+QUICK+START+GUIDE",
        "endpoints": "https://mygroup.atlassian.net/wiki/spaces/CE/cards/pages/168370805/API++Endpoint",
        "landing": "https://mygroup.atlassian.net/wiki/spaces/CE/cards/pages/168508889/COPPER+API+First+Landing+Page",
        "operators": "https://mygroup.atlassian.net/wiki/spaces/CE/cards/pages/168665138/COPPER+SQL+API+Supported+Operators",
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validates critical configuration settings and returns status."""
        missing = []
        
        if not cls.CONFLUENCE_USERNAME:
            missing.append("CONFLUENCE_USERNAME")
        if not cls.CONFLUENCE_API_TOKEN:
            missing.append("CONFLUENCE_API_TOKEN")
        if not cls.STASH_TOKEN:
            missing.append("STASH_TOKEN")
        
        if missing:
            logger.error(f"Missing required configuration: {', '.join(missing)}")
            return False
        
        # Ensure cache directory exists
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        
        return True
    
    @classmethod
    def get_cache_path(cls, name: str) -> str:
        """Returns a path for a cache file."""
        return os.path.join(cls.CACHE_DIR, f"{name}.json")

# =============================================================================
# Content Extraction and Processing
# =============================================================================

class ContentExtractor:
    """Enhanced extractor for Confluence HTML content."""
    
    @staticmethod
    @lru_cache(maxsize=ConfigManager.CACHE_SIZE)
    def extract_content_from_html(html_content: str, title: str = "") -> Dict[str, Any]:
        """
        Extract text, tables, and image contexts from HTML content.
        This method is cached to improve performance for repeated access.
        
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
                caption = table.find('caption')
                table_title = caption.text.strip() if caption else f"Table [{i+1}]"
                
                # Get headers
                headers = []
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
                
                # If no headers in Thead, try getting from first row
                if not headers:
                    first_row = table.find('tr')
                    if first_row:
                        # Check if it looks like a header row (has th elements or all cells look like headers)
                        first_row_cells = first_row.find_all(['th', 'td'])
                        has_th = first_row.find('th') is not None
                        if has_th:
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
                                col_widths[i] = max(col_widths[i], len(cell))
                
                    # Format header row
                    header_row = "| " + " | ".join([h.ljust(col_widths[i]) for i, h in enumerate(headers)]) + " |"
                    separator = "|" + "".join(["-" * (col_widths[i] + 2) + "|" for i, h in enumerate(headers)])
                    table_text += header_row + "\n" + separator + "\n"
                    
                    # Format data rows
                    for row in rows:
                        if len(row) < len(headers):
                            row.extend([""] * (len(headers) - len(row)))
                        row_text = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)]) + " |"
                        table_text += row_text + "\n"
                elif rows:  # Table with no headers
                    # Calculate column width
                    max_cols = max(len(row) for row in rows)
                    col_widths = [0] * max_cols
                    for row in rows:
                        for i, cell in enumerate(row):
                            if i < max_cols:
                                col_widths[i] = max(col_widths[i], len(cell))
                    
                    # Format rows
                    for row in rows:
                        if len(row) < max_cols:
                            row.extend([""] * (max_cols - len(row)))
                        row_text = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)]) + " |"
                        table_text += row_text + "\n"
                
                # Extract structured table data for programmatic use
                structured_table = {
                    "title": table_title,
                    "headers": headers,
                    "rows": rows
                }
                
                if len(table_text) > 1:  # Only add tables with content
                    tables.append({
                        "text": table_text,
                        "data": structured_table
                    })
            
            # Extract images with improved context
            images = []
            for img in soup.find_all('img'):
                # Get image attributes
                alt_text = img.get('alt', '').strip()
                title = img.get('title', '').strip()
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
                    prev_elems = list(img.previous_siblings)[:3]
                    next_elems = list(img.next_siblings)[:3]
                    
                    for elem in prev_elems:
                        if hasattr(elem, 'text') and elem.text.strip():
                            if len(elem.text.strip()) < 200:  # Only short contexts
                                context = f"Previous content: {elem.text.strip()}"
                                break
                
                # Construct meaningful image description
                desc = alt_text or title or "Image"
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
                        if cls.startswith('language-'):
                            lang = cls.replace('language-', '')
                            break
                            
                    code_content = code.text.strip()
                    if lang:
                        code_blocks.append({
                            "language": lang,
                            "content": code_content,
                            "formatted": f"```{lang}\n{code_content}\n```"
                        })
                    else:
                        code_blocks.append({
                            "language": "text",
                            "content": code_content,
                            "formatted": f"```\n{code_content}\n```"
                        })
                else:
                    # Pre without code tag
                    code_blocks.append({
                        "language": "text",
                        "content": pre.text.strip(),
                        "formatted": f"```\n{pre.text.strip()}\n```"
                    })
            
            # Extract any important structured content
            structured_content = []
            for div in soup.find_all(['div', 'section']):
                # Look for common Confluence structured content classes
                class_value = div.get('class', [])
                classnames = ['panel', 'info', 'note', 'warning', 'callout', 'aui-message']
                
                matched = any(classname in classnames for classname in class_value)
                
                if matched:
                    title_elem = div.find(['h1', 'h2', 'h3', 'span', 'strong'])
                    title_text = title_elem.text.strip() if title_elem else "NOTE"
                    content = div.text.strip()
                    if title_elem and title_elem.text in content:
                        content = content.replace(title_elem.text, "", 1).strip()
                    structured_content.append(f"**{title_text}** -- {content}")
            
            # Look for specific mapping tables that might indicate view-to-API mapping
            mapping_tables = []
            for table_data in tables:
                table = table_data["data"]
                headers = [h.lower() for h in table["headers"]] if table["headers"] else []
                
                # Check if this looks like a mapping table
                mapping_indicators = [
                    "view", "api", "endpoint", "attribute", "column", "field", 
                    "table", "map", "mapping", "copper"
                ]
                
                if any(indicator in " ".join(headers).lower() for indicator in mapping_indicators):
                    mapping_tables.append(table)
            
            return {
                "text": "\n\n".join(text_content),
                "tables": tables,
                "mapping_tables": mapping_tables,
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
                "mapping_tables": [],
                "images": [],
                "code_blocks": [],
                "structured_content": []
            }
    
    @staticmethod
    def format_for_context(extracted_content: Dict[str, Any], title: str = "") -> str:
        """
        Format the extracted content for use as context with improved structure.
        
        Args:
            extracted_content: The dictionary of extracted content
            title: The title of the page
            
        Returns:
            Formatted string containing all the content
        """
        sections = []
        
        if title:
            sections.append(f"# {title}")
        
        if extracted_content["text"]:
            sections.append(extracted_content["text"])
        
        if extracted_content["tables"]:
            sections.append("\n\n### Tables:\n")
            for table in extracted_content["tables"]:
                sections.append(table["text"])
        
        if extracted_content["code_blocks"]:
            sections.append("\n\n### Code Examples:\n")
            for code_block in extracted_content["code_blocks"]:
                sections.append(code_block["formatted"])
        
        if extracted_content["structured_content"]:
            sections.append("\n\n### Important Notes:\n")
            sections.extend(extracted_content["structured_content"])
        
        if extracted_content["images"]:
            sections.append("\n\n### Image Information:\n")
            sections.extend(extracted_content["images"])
        
        return "\n\n".join(sections)

    @staticmethod
    def extract_mapping_information(extracted_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract potential view-to-API mapping information from the content.
        
        Args:
            extracted_content: The extracted content dictionary
            
        Returns:
            List of mapping dictionaries
        """
        mappings = []
        
        # Process mapping tables if available
        for table in extracted_content.get("mapping_tables", []):
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            if not headers or not rows:
                continue
                
            # Normalize headers for consistent processing
            norm_headers = [h.lower().strip() for h in headers]
            
            # Identify important columns
            view_col = None
            api_col = None
            endpoint_col = None
            attribute_col = None
            
            for i, header in enumerate(norm_headers):
                if any(term in header for term in ["view", "source"]):
                    view_col = i
                elif "api" in header and any(term in header for term in ["endpoint", "path"]):
                    api_col = i
                elif any(term in header for term in ["endpoint", "service"]):
                    endpoint_col = i
                elif any(term in header for term in ["attribute", "field", "column", "property"]):
                    attribute_col = i
            
            # If we couldn't identify crucial columns, this might not be a mapping table
            if view_col is None or (api_col is None and endpoint_col is None):
                continue
                
            # Process each row as a mapping
            for row in rows:
                if len(row) != len(headers):
                    continue  # Skip malformed rows
                    
                mapping = {}
                
                # Extract view information
                if view_col is not None and view_col < len(row):
                    mapping["view"] = row[view_col].strip()
                    
                # Extract API endpoint
                if api_col is not None and api_col < len(row):
                    mapping["api_endpoint"] = row[api_col].strip()
                elif endpoint_col is not None and endpoint_col < len(row):
                    mapping["api_endpoint"] = row[endpoint_col].strip()
                    
                # Extract attribute mapping
                if attribute_col is not None and attribute_col < len(row):
                    mapping["attribute"] = row[attribute_col].strip()
                    
                # Include all fields for completeness
                mapping["all_fields"] = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                
                if mapping.get("view") and mapping.get("api_endpoint"):
                    mappings.append(mapping)
        
        # Also look for mapping patterns in text
        text = extracted_content.get("text", "")
        
        # Look for patterns like "The view X maps to API endpoint Y"
        view_to_api_patterns = [
            r'(?:view|table)\s+["\']?([A-Za-z0-9_]+)["\']?\s+(?:map|correspond)s?\s+to\s+(?:api|endpoint)\s+["\']?([A-Za-z0-9_/]+)["\']?',
            r'(?:api|endpoint)\s+["\']?([A-Za-z0-9_/]+)["\']?\s+(?:map|correspond)s?\s+to\s+(?:view|table)\s+["\']?([A-Za-z0-9_]+)["\']?'
        ]
        
        for pattern in view_to_api_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if pattern.startswith('(?:view|table)'):
                    mapping = {
                        "view": match.group(1),
                        "api_endpoint": match.group(2),
                        "source": "text_pattern"
                    }
                else:
                    mapping = {
                        "view": match.group(2),
                        "api_endpoint": match.group(1),
                        "source": "text_pattern"
                    }
                mappings.append(mapping)
        
        return mappings

# =============================================================================
# Confluence Client
# =============================================================================

class ConfluenceClient:
    """Enhanced client for Confluence REST API with multi-space support."""
    
    def __init__(self, base_url: str, username: str, api_token: str):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.timeout = 30
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize space pages cache
        self.space_pages = {}
        
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self) -> bool:
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
    
    @lru_cache(maxsize=ConfigManager.CACHE_SIZE)
    def get_cached_request(self, url: str, params_str: str) -> Optional[str]:
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
    
    def get_all_pages_in_space(self, space_key: str, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Get all pages in a Confluence space using efficient pagination.
        
        Args:
            space_key: The space key to get all pages from
            batch_size: Number of results per request (max 100)
            
        Returns:
            List of page objects with basic information
        """
        # Return from internal cache if available
        if space_key in self.space_pages:
            return self.space_pages[space_key]
            
        all_pages = []
        start = 0
        has_more = True
        
        logger.info(f"Fetching all pages from space: {space_key}")
        
        # Check if we have cached results
        cache_path = ConfigManager.get_cache_path(f"space_pages_{space_key}")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('space_key') == space_key:
                        logger.info(f"Using cached page list from {cache_path}")
                        self.space_pages[space_key] = cached_data.get('pages', [])
                        return self.space_pages[space_key]
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
        
        # Store in instance cache
        self.space_pages[space_key] = all_pages
        
        logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
        return all_pages
    
    def get_page_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the content of a page in a suitable format for NLP.
        
        Args:
            page_id: The ID of the page
            
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
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "space_key": page.get("_expandable", {}).get("space", "").split("/")[-1],
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
                "labels": [label.get('name') for label in page.get('metadata', {}).get('labels', {}).get('results', [])]
            }
            
            # Get raw content
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process with our advanced content extractor
            extracted_content = ContentExtractor.extract_content_from_html(html_content, page.get("title", ""))
            formatted_content = ContentExtractor.format_for_context(extracted_content, page.get("title", ""))
            
            # Extract any mapping information
            mapping_info = ContentExtractor.extract_mapping_information(extracted_content)
            
            result = {
                "metadata": metadata,
                "content": extracted_content,
                "formatted_content": formatted_content,
                "mapping_info": mapping_info,
                "raw_html": html_content
            }
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_content_by_id(self, content_id: str, expand: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get content by ID with optional expansion parameters."""
        try:
            params = {}
            if expand:
                params["expand"] = expand
                
            # Convert params to string for cache key
            params_str = json.dumps(params, sort_keys=True)
            
            # Try to get from cache first
            response_text = self.get_cached_request(f"{self.api_url}/content/{content_id}", params_str)
            
            if not response_text:
                logger.warning(f"Empty response received when retrieving content ID: {content_id}")
                return None
                
            content = json.loads(response_text)
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting content by ID {content_id}: {str(e)}")
            return None
    
    def search_content(self, query: str, space_keys: List[str]) -> List[Dict[str, Any]]:
        """
        Search content across multiple Confluence spaces.
        
        Args:
            query: The search query
            space_keys: List of space keys to search in
            
        Returns:
            List of relevant pages with content
        """
        all_results = []
        
        # Search in each specified space
        for space_key in space_keys:
            space_results = self.search_space_content(query, space_key)
            all_results.extend(space_results)
        
        # Deduplicate results (in case a page appears in multiple searches)
        unique_results = {}
        for result in all_results:
            page_id = result[0].get('id')
            if page_id not in unique_results or unique_results[page_id][2] < result[2]:
                unique_results[page_id] = result
        
        # Sort by relevance score and limit results
        sorted_results = sorted(unique_results.values(), key=lambda x: x[2], reverse=True)[:10]
        
        logger.info(f"Found {len(sorted_results)} relevant pages across {len(space_keys)} spaces")
        return sorted_results
    
    def search_space_content(self, query: str, space_key: str) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Search content in the specified Confluence space.
        
        Args:
            query: The search query
            space_key: The space to search in
            
        Returns:
            List of tuples (page metadata, page content, relevance score)
        """
        # Ensure we have loaded space pages
        if space_key not in self.space_pages:
            self.space_pages[space_key] = self.get_all_pages_in_space(space_key)
            
        if not self.space_pages.get(space_key):
            logger.warning(f"No pages found in space: {space_key}")
            return []
            
        # Step 1: Score pages based on title and perform initial filtering
        query_words = query.lower().split()
        candidates = []
        
        # Add some domain-specific terms for better filtering
        domain_terms = ["database", "view", "api", "endpoint", "rest", "mapping", 
                       "copper", "json", "sql", "attribute", "table", "field", "column"]
        
        # Initial scoring based on title matches
        for page in self.space_pages[space_key]:
            title = page.get('title', '').lower()
            # Check match score in title
            title_score = 0
            for word in query_words:
                if word in title:
                    title_score += 3  # Higher weight for title matches
                    
            # If title has matches, or it mentions key terms, include as candidate
            if title_score > 0 or any(term in title for term in domain_terms):
                candidates.append((page, title_score))
                
        # Sort candidates by initial score and take top 20 for detailed analysis
        top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
        
        logger.info(f"Selected {len(top_candidates)} candidate pages for detailed analysis in space {space_key}")
        
        # Fetch content for candidate pages in parallel
        page_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=ConfigManager.MAX_WORKERS) as executor:
            futures = {executor.submit(self.get_page_content, page[0]['id']): page for page in top_candidates}
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                try:
                    content = future.result()
                    if content:
                        page_data.append((candidate[0], content, candidate[1]))
                except Exception as e:
                    logger.error(f"Error fetching content for page {candidate[0].get('id')}: {str(e)}")
        
        # If no candidates found by title, load 10 most recently updated pages
        if not page_data:
            logger.info(f"No candidates found by title match, using recent pages from space {space_key}")
            recent_pages = sorted(
                self.space_pages[space_key], 
                key=lambda x: x.get('history', {}).get('lastUpdated', '2000-01-01'),
                reverse=True
            )[:10]
            
            page_dict = {p['id']: p for p in recent_pages}
            
            # Fetch content for these pages
            for page_id in page_dict:
                content = self.get_page_content(page_id)
                if content:
                    page_data.append((page_dict[page_id], content, 0))
        
        # Step 2: Detailed relevance scoring with content
        scored_pages = []
        for page, content, initial_score in page_data:
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
            
            # Extra bonus for mention of COPPER in title
            if "copper" in page.get('title', '').lower():
                score += 5
                
            # Extra bonus for pages with mapping information
            if content.get('mapping_info'):
                score += len(content['mapping_info']) * 2
                
            scored_pages.append((page, content, score))
            
        # Sort by score and return
        return sorted(scored_pages, key=lambda x: x[2], reverse=True)

    def get_content_by_url(self, confluence_url: str) -> Optional[Dict[str, Any]]:
        """
        Get page content from a Confluence URL.
        
        Args:
            confluence_url: Full URL to a Confluence page
            
        Returns:
            Page content if found
        """
        try:
            # Try to extract page ID from URL
            page_id = None
            
            # Pattern for modern Confluence URLs like /wiki/spaces/SPACE/pages/123456789
            match = re.search(r'/pages/(\d+)', confluence_url)
            if match:
                page_id = match.group(1)
            
            # Pattern for old style Confluence URLs
            if not page_id:
                match = re.search(r'pageId=(\d+)', confluence_url)
                if match:
                    page_id = match.group(1)
            
            if not page_id:
                logger.warning(f"Could not extract page ID from URL: {confluence_url}")
                return None
                
            # Get the page content using the ID
            return self.get_page_content(page_id)
            
        except Exception as e:
            logger.error(f"Error getting content from URL {confluence_url}: {str(e)}")
            return None

    def get_important_pages(self, important_pages: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Get content for a predefined set of important pages.
        
        Args:
            important_pages: Dictionary of {key: confluence_url}
            
        Returns:
            Dictionary of {key: page_content}
        """
        results = {}
        
        for key, url in important_pages.items():
            logger.info(f"Fetching important page: {key} ({url})")
            content = self.get_content_by_url(url)
            if content:
                results[key] = content
                logger.info(f"Successfully fetched important page: {key}")
            else:
                logger.warning(f"Failed to fetch important page: {key}")
                
        return results

# =============================================================================
# Stash/Bitbucket Client
# =============================================================================

class StashClient:
    """Client for Bitbucket/Stash REST API operations focused on retrieving COPPER view definitions."""
    
    def __init__(self, base_url: str, api_token: str, project_key: str, repo_slug: str, directory_path: str = ""):
        """
        Initialize the Stash client with authentication details.
        
        Args:
            base_url: Base URL of the Stash/Bitbucket server
            api_token: API token for authentication
            project_key: Project key
            repo_slug: Repository slug
            directory_path: Path to the directory containing view definitions
        """
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.project_key = project_key
        self.repo_slug = repo_slug
        self.directory_path = directory_path
        
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
        self.session = requests.Session()
        self.timeout = 30
        
        # Cache for API responses
        self._file_listing_cache = None
        self._file_content_cache = {}
        
        logger.info(f"Initialized Stash client for {self.base_url}, project {self.project_key}, repo {self.repo_slug}")
    
    def test_connection(self) -> bool:
        """Test the connection to Stash/Bitbucket API."""
        try:
            logger.info("Testing connection to Stash API...")
            url = f"{self.base_url}/rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}"
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info("Connection to Stash successful!")
                return True
            else:
                logger.warning("Empty response received during connection test")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_file_listing(self, path: str = "") -> Optional[List[Dict[str, Any]]]:
        """
        Get listing of files in the specified directory.
        
        Args:
            path: Path within the repository
            
        Returns:
            List of file information dictionaries
        """
        # Use cached listing if available
        if self._file_listing_cache is not None:
            return self._file_listing_cache
            
        # If directory path is specified in the instance, append it to the path
        if self.directory_path and not path:
            path = self.directory_path
            
        try:
            url = f"{self.base_url}/rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}/browse"
            params = {"limit": 1000}  # Max limit
            if path:
                params["path"] = path
                
            logger.info(f"Fetching file listing from Stash: {path}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
                verify=False
            )
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract file information
            files = []
            for child in data.get("children", {}).get("values", []):
                # Only include files, not directories
                if child.get("type") == "FILE":
                    file_info = {
                        "path": child.get("path", {}).get("toString", ""),
                        "name": child.get("path", {}).get("name", ""),
                        "type": child.get("type"),
                        "size": child.get("size", 0)
                    }
                    files.append(file_info)
            
            logger.info(f"Found {len(files)} files in {path}")
            
            # Cache the results
            self._file_listing_cache = files
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting file listing: {str(e)}")
            return None
    
    def get_file_content(self, filename: str) -> Optional[str]:
        """
        Get content of a file.
        
        Args:
            filename: Name or path of the file
            
        Returns:
            Content of the file as string
        """
        # Use cached content if available
        if filename in self._file_content_cache:
            return self._file_content_cache[filename]
            
        # Determine the full path
        file_path = filename
        if self.directory_path and not filename.startswith(self.directory_path):
            file_path = f"{self.directory_path}/{filename}"
            
        try:
            url = f"{self.base_url}/rest/api/1.0/projects/{self.project_key}/repos/{self.repo_slug}/raw/{file_path}"
            
            logger.info(f"Fetching file content from Stash: {file_path}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            response.raise_for_status()
            
            content = response.text
            
            # Cache the content
            self._file_content_cache[filename] = content
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting file content for {file_path}: {str(e)}")
            return None
    
    def find_view_definition(self, view_name: str) -> Optional[str]:
        """
        Find the definition of a specific database view.
        
        Args:
            view_name: Name of the view to find
            
        Returns:
            SQL definition of the view if found
        """
        # Normalize view name for comparison
        normalized_name = view_name.lower().strip()
        
        # Get listing of files in the directory
        files = self.get_file_listing()
        if not files:
            logger.warning(f"No files found in repository when searching for view {view_name}")
            return None
            
        # Look for exact filename match
        exact_matches = [f for f in files if f.get("name", "").lower() == f"{normalized_name}.sql"]
        if exact_matches:
            logger.info(f"Found exact match for view {view_name}: {exact_matches[0]['path']}")
            return self.get_file_content(exact_matches[0]["path"])
            
        # Look for partial filename matches
        partial_matches = [f for f in files if normalized_name in f.get("name", "").lower()]
        if partial_matches:
            logger.info(f"Found partial match for view {view_name}: {partial_matches[0]['path']}")
            return self.get_file_content(partial_matches[0]["path"])
            
        # If no filename matches, search contents of files
        logger.info(f"No filename matches for view {view_name}, searching through file contents")
        
        # Limit to reasonable number of files to search
        files_to_search = min(50, len(files))
        for i in range(files_to_search):
            content = self.get_file_content(files[i]["path"])
            if content:
                # Look for create/replace view statements
                patterns = [
                    r'create\s+(?:or\s+replace\s+)?view\s+([^\s\(]+)\s+as',
                    r'replace\s+view\s+([^\s\(]+)\s+as'
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        found_view = match.group(1).strip().lower().replace('"', '').replace('[', '').replace(']', '')
                        if found_view == normalized_name:
                            logger.info(f"Found view definition in content of {files[i]['path']}")
                            return content
        
        logger.warning(f"No view definition found for {view_name}")
        return None

# =============================================================================
# View Analysis
# =============================================================================

class ViewAnalyzer:
    """Analyzer for SQL view definitions."""
    
    @staticmethod
    def parse_sql(sql_code: str) -> Dict[str, Any]:
        """
        Parse SQL view definition to extract key components.
        
        Args:
            sql_code: SQL code defining the view
            
        Returns:
            Dictionary of parsed components
        """
        try:
            # Parse the SQL statement
            parsed = sqlparse.parse(sql_code)
            if not parsed:
                logger.warning("Failed to parse SQL code")
                return {"success": False, "error": "Empty or invalid SQL"}
                
            # Extract view name
            view_name_pattern = r'create\s+(?:or\s+replace\s+)?view\s+([^\s\(]+)\s+as'
            view_name_match = re.search(view_name_pattern, sql_code, re.IGNORECASE)
            view_name = view_name_match.group(1) if view_name_match else "Unknown View"
            view_name = view_name.strip().replace('"', '').replace('[', '').replace(']', '')
            
            # Extract referenced tables
            referenced_tables = []
            from_pattern = r'from\s+([^\s\,\)]+)'
            join_pattern = r'join\s+([^\s\,\)]+)'
            
            for pattern in [from_pattern, join_pattern]:
                for match in re.finditer(pattern, sql_code, re.IGNORECASE):
                    table = match.group(1).strip().replace('"', '').replace('[', '').replace(']', '')
                    if table.lower() not in ['select', 'where', 'and', 'or', 'on']:
                        referenced_tables.append(table)
            
            # Extract selected columns
            selected_columns = []
            select_pattern = r'select\s+(.*?)\s+from'
            select_match = re.search(select_pattern, sql_code, re.IGNORECASE | re.DOTALL)
            
            if select_match:
                columns_string = select_match.group(1)
                # Handle case of SELECT *
                if '*' in columns_string and len(columns_string.strip()) <= 3:
                    selected_columns = ["*"]
                else:
                    # Split by commas, accounting for nested expressions
                    depth = 0
                    current_column = []
                    for char in columns_string:
                        if char == '(' or char == '[':
                            depth += 1
                        elif char == ')' or char == ']':
                            depth -= 1
                            
                        if char == ',' and depth == 0:
                            selected_columns.append(''.join(current_column).strip())
                            current_column = []
                        else:
                            current_column.append(char)
                            
                    if current_column:
                        selected_columns.append(''.join(current_column).strip())
            
            # Try to extract column aliases
            columns_with_aliases = []
            for col in selected_columns:
                # Extract alias using AS keyword
                as_pattern = r'(.*?)\s+as\s+([^\s,]+)$'
                as_match = re.search(as_pattern, col, re.IGNORECASE)
                
                if as_match:
                    source = as_match.group(1).strip()
                    alias = as_match.group(2).strip().replace('"', '').replace('[', '').replace(']', '')
                    columns_with_aliases.append({"source": source, "alias": alias})
                else:
                    # Check for implicit alias (no AS keyword)
                    if ' ' in col and not re.search(r'[\(\)]', col):
                        parts = col.rsplit(' ', 1)
                        if len(parts) == 2:
                            source = parts[0].strip()
                            alias = parts[1].strip().replace('"', '').replace('[', '').replace(']', '')
                            columns_with_aliases.append({"source": source, "alias": alias})
                        else:
                            columns_with_aliases.append({"source": col, "alias": None})
                    else:
                        # If it's a direct column reference with no alias
                        if '.' in col:
                            alias = col.split('.')[-1].strip().replace('"', '').replace('[', '').replace(']', '')
                        else:
                            alias = col.strip().replace('"', '').replace('[', '').replace(']', '')
                        columns_with_aliases.append({"source": col, "alias": alias})
            
            return {
                "success": True,
                "view_name": view_name,
                "referenced_tables": list(set(referenced_tables)),  # Remove duplicates
                "columns": columns_with_aliases,
                "raw_sql": sql_code
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SQL: {str(e)}")
            return {"success": False, "error": str(e), "raw_sql": sql_code}
    
    @staticmethod
    def get_column_source_info(column: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract source information for a column.
        
        Args:
            column: Column definition dictionary
            
        Returns:
            Source information dictionary
        """
        source = column.get("source", "")
        alias = column.get("alias")
        
        # If it's a direct column reference
        if '.' in source and not re.search(r'[\(\)]', source):
            parts = source.split('.')
            if len(parts) == 2:
                table = parts[0].strip().replace('"', '').replace('[', '').replace(']', '')
                col = parts[1].strip().replace('"', '').replace('[', '').replace(']', '')
                return {
                    "type": "direct",
                    "table": table,
                    "column": col,
                    "alias": alias or col
                }
        
        # If it's a function or complex expression
        if re.search(r'[\(\)]', source):
            # Try to extract any column references within
            col_refs = re.findall(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', source)
            return {
                "type": "expression",
                "expression": source,
                "referenced_columns": [{"table": ref[0], "column": ref[1]} for ref in col_refs],
                "alias": alias or "expression"
            }
        
        # If it's a simple column with no table qualification
        return {
            "type": "simple",
            "column": source.strip().replace('"', '').replace('[', '').replace(']', ''),
            "alias": alias or source
        }

# =============================================================================
# API Mapping Engine
# =============================================================================

class ApiMappingEngine:
    """Engine for mapping database views to API endpoints."""
    
    def __init__(self, confluence_client: ConfluenceClient):
        """
        Initialize the mapping engine with a Confluence client.
        
        Args:
            confluence_client: Initialized Confluence client
        """
        self.confluence = confluence_client
        self.mapping_cache = {}
        self.important_pages = {}
        self.loaded = False
        
    def load_mapping_knowledge(self) -> None:
        """Load important pages and mapping knowledge from Confluence."""
        if self.loaded:
            return
            
        logger.info("Loading API mapping knowledge from Confluence...")
        
        # Load important pages
        self.important_pages = self.confluence.get_important_pages(ConfigManager.IMPORTANT_PAGES)
        
        # Extract mapping information from the pages
        all_mappings = []
        for key, page in self.important_pages.items():
            if page and 'mapping_info' in page:
                all_mappings.extend(page['mapping_info'])
                
        logger.info(f"Loaded {len(all_mappings)} mapping entries from important pages")
        
        # Index mappings by view for fast lookup
        for mapping in all_mappings:
            view = mapping.get('view', '').lower()
            if view:
                if view not in self.mapping_cache:
                    self.mapping_cache[view] = []
                self.mapping_cache[view].append(mapping)
                
        self.loaded = True
    
    def get_mapping_for_view(self, view_name: str) -> List[Dict[str, Any]]:
        """
        Get API mappings for a specific view.
        
        Args:
            view_name: Name of the view
            
        Returns:
            List of mapping entries
        """
        self.load_mapping_knowledge()
        
        # Normalize view name for lookup
        view_name_lower = view_name.lower()
        
        # Direct lookup from cache
        if view_name_lower in self.mapping_cache:
            return self.mapping_cache[view_name_lower]
            
        # Try fuzzy matching
        for cached_view, mappings in self.mapping_cache.items():
            # Check if view name is contained in cached view name or vice versa
            if view_name_lower in cached_view or cached_view in view_name_lower:
                logger.info(f"Found fuzzy match for view {view_name}: {cached_view}")
                return mappings
                
        # If no match found, search Confluence
        logger.info(f"No cached mapping for view {view_name}, searching Confluence...")
        search_results = self.confluence.search_content(
            f"view {view_name} mapping API endpoint", 
            ConfigManager.CONFLUENCE_SPACES
        )
        
        # Extract mapping info from search results
        mappings = []
        for _, content, _ in search_results:
            if 'mapping_info' in content:
                # Filter to mappings relevant to this view
                for mapping in content['mapping_info']:
                    mapping_view = mapping.get('view', '').lower()
                    if mapping_view and (mapping_view == view_name_lower or 
                                         view_name_lower in mapping_view or 
                                         mapping_view in view_name_lower):
                        mappings.append(mapping)
        
        # Cache the results
        if mappings:
            self.mapping_cache[view_name_lower] = mappings
            
        return mappings
    
    def map_view_to_api(self, view_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a parsed SQL view to API endpoints.
        
        Args:
            view_info: Parsed view information (from ViewAnalyzer)
            
        Returns:
            Mapping details dictionary
        """
        if not view_info.get("success", False):
            return {"success": False, "error": view_info.get("error", "Unknown error in view info")}
            
        view_name = view_info.get("view_name", "")
        if not view_name:
            return {"success": False, "error": "View name not found in view info"}
            
        # Get mapping information for this view
        mappings = self.get_mapping_for_view(view_name)
        
        # Extract columns from view info
        columns = view_info.get("columns", [])
        column_aliases = {col.get("alias"): col for col in columns if col.get("alias")}
        
        # Create the mapping result
        result = {
            "success": True,
            "view_name": view_name,
            "api_mapping": {
                "endpoints": [],
                "columns": []
            },
            "referenced_tables": view_info.get("referenced_tables", []),
            "raw_mappings": mappings
        }
        
        # Process mappings for endpoints
        endpoints = set()
        for mapping in mappings:
            endpoint = mapping.get("api_endpoint")
            if endpoint:
                endpoints.add(endpoint)
                
        result["api_mapping"]["endpoints"] = list(endpoints)
        
        # Process column mappings
        for col in columns:
            alias = col.get("alias")
            if not alias:
                continue
                
            # Find mapping for this column
            column_mapping = {"view_column": alias, "mapped": False}
            
            for mapping in mappings:
                if "all_fields" in mapping:
                    # Look for this column name in the mapping fields
                    for field_name, field_value in mapping["all_fields"].items():
                        # Check if field name looks like a column header (view column or API attribute)
                        if re.search(r'column|field|attribute|view|api', field_name, re.IGNORECASE):
                            # Check if value matches our column alias
                            if field_value.lower() == alias.lower():
                                # Find corresponding API attribute
                                for api_field_name, api_field_value in mapping["all_fields"].items():
                                    if re.search(r'api|endpoint|attribute', api_field_name, re.IGNORECASE) and api_field_name != field_name:
                                        column_mapping["api_attribute"] = api_field_value
                                        column_mapping["mapped"] = True
                                        column_mapping["endpoint"] = mapping.get("api_endpoint", "Unknown")
                                        break
                
                # Also check attribute field directly
                if "attribute" in mapping and mapping["attribute"].lower() == alias.lower():
                    column_mapping["api_attribute"] = mapping["attribute"]
                    column_mapping["mapped"] = True
                    column_mapping["endpoint"] = mapping.get("api_endpoint", "Unknown")
            
            # Add source information
            column_mapping["source_info"] = ViewAnalyzer.get_column_source_info(col)
            
            result["api_mapping"]["columns"].append(column_mapping)
        
        return result
    
    def generate_mapping_report(self, mapping_result: Dict[str, Any]) -> str:
        """
        Generate a user-friendly mapping report.
        
        Args:
            mapping_result: Result from map_view_to_api
            
        Returns:
            Formatted report text
        """
        if not mapping_result.get("success", False):
            return f"Error generating mapping: {mapping_result.get('error', 'Unknown error')}"
            
        view_name = mapping_result.get("view_name", "Unknown View")
        api_mapping = mapping_result.get("api_mapping", {})
        endpoints = api_mapping.get("endpoints", [])
        columns = api_mapping.get("columns", [])
        
        # Generate report
        report = [f"# Mapping Report for View: {view_name}\n"]
        
        # Endpoints section
        report.append("## API Endpoints\n")
        if endpoints:
            for endpoint in endpoints:
                report.append(f"* {endpoint}")
        else:
            report.append("No API endpoints found for this view.")
            
        report.append("\n## Column Mappings\n")
        
        # Column mappings section
        if columns:
            table_rows = []
            table_rows.append("| View Column | API Attribute | Endpoint | Mapped? | Source Info |")
            table_rows.append("|------------|--------------|---------|---------|------------|")
            
            for col in columns:
                view_col = col.get("view_column", "Unknown")
                api_attr = col.get("api_attribute", "Not mapped")
                endpoint = col.get("endpoint", "N/A")
                mapped = "✅" if col.get("mapped", False) else "❌"
                
                # Format source info
                source_info = col.get("source_info", {})
                if source_info.get("type") == "direct":
                    source = f"{source_info.get('table', '')}.{source_info.get('column', '')}"
                elif source_info.get("type") == "expression":
                    source = "Expression"
                else:
                    source = source_info.get("column", "Unknown")
                    
                table_rows.append(f"| {view_col} | {api_attr} | {endpoint} | {mapped} | {source} |")
                
            report.extend(table_rows)
        else:
            report.append("No column mappings available.")
            
        # Referenced tables section
        report.append("\n## Referenced Tables\n")
        ref_tables = mapping_result.get("referenced_tables", [])
        if ref_tables:
            for table in ref_tables:
                report.append(f"* {table}")
        else:
            report.append("No referenced tables identified.")
            
        # Orchestration section
        report.append("\n## Orchestration Info\n")
        
        # Determine if orchestration is needed
        endpoints_count = len(endpoints)
        if endpoints_count > 1:
            report.append("**Orchestration Required**: Yes - multiple API endpoints need to be called and results combined.")
            report.append("\nSuggested orchestration steps:")
            for i, endpoint in enumerate(endpoints):
                report.append(f"{i+1}. Call endpoint: {endpoint}")
            report.append(f"{len(endpoints)+1}. Combine results based on common keys")
        elif endpoints_count == 1:
            report.append("**Orchestration Required**: No - a single API endpoint can satisfy this view.")
        else:
            report.append("**Orchestration Required**: Unknown - no endpoints identified.")
            
        return "\n".join(report)

# =============================================================================
# Gemini Assistant
# =============================================================================

class GeminiAssistant:
    """Gemini AI assistant for COPPER view to API mapping."""
    
    def __init__(self):
        """Initialize the Gemini assistant."""
        self.initialized_flag = False
        
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini AI modules not available. Running in limited mode.")
            return
            
        try:
            # Initialize Vertex AI
            vertexai.init(project=ConfigManager.PROJECT_ID, location=ConfigManager.REGION)
            self.model = GenerativeModel(ConfigManager.MODEL_NAME)
            self.initialized_flag = True
            logger.info(f"Initialized Gemini Assistant with model: {ConfigManager.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Error initializing Gemini Assistant: {str(e)}")
    
    def initialized(self) -> bool:
        """Check if Gemini assistant is properly initialized."""
        return self.initialized_flag
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using Gemini based on the prompt and context.
        
        Args:
            prompt: The user's question or prompt
            context: Context information
            
        Returns:
            The generated response
        """
        if not self.initialized_flag:
            return "Gemini AI is not available. Please check your configuration."
            
        logger.info(f"Generating response for prompt: {prompt}")
        
        try:
            # Create a system prompt with guidelines for response
            system_prompt = """
            You are the friendly COPPER Assistant, an expert on mapping database views to COPPER REST APIs.
            
            Your personality:
            - Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
            - Explain technical concepts in plain language, as if speaking to a colleague
            - Prioritize clarity and conciseness - avoid jargon when possible
            - Add occasional light humor where appropriate to make the conversation engaging
            - Be concise but thorough - focus on answering the question directly first, then add helpful context
            
            Your expertise:
            - Deep knowledge of the COPPER database system, its views, and corresponding API endpoints
            - Experience with database to API mapping patterns and best practices
            - Awareness of how applications integrate with COPPER's REST APIs
            - Expert in interpreting table structures, field mappings, and API parameters
            - Understanding CME's specific systems and data model
            
            When answering:
            1. Directly address the user's question first
            2. Provide practical, actionable information when possible
            3. Format tables and structured data to make it easier to understand
            4. Use bullet points or numbered lists for steps, multiple items
            5. Reference specific examples from the documentation when available
            6. Acknowledge any limitations in the available information
            
            If the context includes mapping information, refer to it specifically. If the question is about a view or API that isn't in your context, explain that you don't have that specific information and suggest where they might look.
            
            Remember to maintain a balance between being friendly and professional - you're a helpful colleague, not a formal technical document.
            """
            
            # Craft the full prompt with context
            full_prompt = system_prompt + "\n\n"
            
            # Trim context if needed while preserving most relevant parts
            if context is not None and len(context) > 10000:
                # First 3000 characters often contain important intro information
                intro = context[:3000]
                # Last 7000 characters might contain conclusions or summaries
                outro = context[-7000:]
                context = intro + "\n\n[...content trimmed for length...]\n\n" + outro
                
            if context:
                full_prompt += "CONTEXT INFORMATION:\n" + context + "\n\n"
                
            full_prompt += "USER QUESTION: " + prompt + "\n\nResponse:"
            
            # Configure generation parameters
            temperature = 0.2  # Lower temperature for more factual responses
            
            # Generate the response
            response = self.model.generate_content(
                full_prompt,
                generation_config=GenerationConfig(
                    temperature=temperature,
                )
            )
            
            if response.candidates and response.candidates[0].text:
                response_text = response.candidates[0].text.strip()
                logger.info(f"Successfully generated response: {response_text[:100]}...")
                return response_text
            else:
                logger.warning("No response generated from Gemini")
                return "I couldn't find a specific answer to that question in our documentation. Could you try rephrasing, or maybe I can help you find the right documentation to look at?"
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I ran into a technical issue while looking that up. Error: {str(e)}"
    
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate an answer to a user question with appropriate context.
        
        Args:
            question: The user's question
            context: Dictionary containing context information
            
        Returns:
            Response text
        """
        # Format the context based on its type
        formatted_context = ""
        
        # Mapping result context
        if "mapping_result" in context:
            mapping_result = context["mapping_result"]
            if mapping_result.get("success", False):
                # Include the mapping report
                report = context.get("mapping_report", "")
                if report:
                    formatted_context += report + "\n\n"
                
                # Include raw SQL if available
                if "raw_sql" in mapping_result:
                    formatted_context += "SQL DEFINITION:\n```sql\n" + mapping_result["raw_sql"] + "\n```\n\n"
        
        # Confluence search results context
        if "search_results" in context:
            search_results = context["search_results"]
            formatted_context += "INFORMATION FROM DOCUMENTATION:\n\n"
            
            for i, (page, content, _) in enumerate(search_results[:3]):  # Limit to top 3 results
                title = page.get("title", "Untitled Page")
                formatted_context += f"--- Document {i+1}: {title} ---\n"
                
                # Include formatted content
                if "formatted_content" in content:
                    # Try to keep only the most relevant parts
                    text = content["formatted_content"]
                    if len(text) > 2000:
                        text = text[:2000] + "...[content trimmed]..."
                    formatted_context += text + "\n\n"
                
                # Include mapping information if available
                if "mapping_info" in content and content["mapping_info"]:
                    formatted_context += f"MAPPING INFORMATION FROM {title}:\n"
                    for mapping in content["mapping_info"][:5]:  # Limit to first 5 mappings
                        formatted_context += f"* View: {mapping.get('view', 'Unknown')}, API: {mapping.get('api_endpoint', 'Unknown')}\n"
                    formatted_context += "\n"
        
        # Special context for specific question types
        if "api_docs" in context:
            formatted_context += "API DOCUMENTATION:\n" + context["api_docs"] + "\n\n"
            
        if "faq" in context:
            formatted_context += "FAQ INFORMATION:\n" + context["faq"] + "\n\n"
        
        logger.info(f"Answering question with context length: {len(formatted_context)}")
        return self.generate_response(question, formatted_context)

# =============================================================================
# COPPER View-to-API Mapper
# =============================================================================

class CopperViewApiMapper:
    """Main class for the COPPER View-to-API Mapper application."""
    
    def __init__(self):
        """Initialize the COPPER View-to-API Mapper."""
        # Validate configuration
        logger.info("Initializing COPPER View-to-API Mapper")
        if not ConfigManager.validate_config():
            logger.error("Configuration validation failed")
            self.initialized = False
            return
            
        # Initialize components
        try:
            # Confluence client for documentation
            self.confluence = ConfluenceClient(
                ConfigManager.CONFLUENCE_URL,
                ConfigManager.CONFLUENCE_USERNAME,
                ConfigManager.CONFLUENCE_API_TOKEN
            )
            
            # Stash client for view definitions
            self.stash = StashClient(
                ConfigManager.STASH_URL,
                ConfigManager.STASH_TOKEN,
                ConfigManager.STASH_PROJECT_KEY,
                ConfigManager.STASH_REPO_SLUG,
                ConfigManager.STASH_VIEW_PATH
            )
            
            # API mapping engine
            self.mapper = ApiMappingEngine(self.confluence)
            
            # Gemini assistant
            self.assistant = GeminiAssistant()
            
            # Test connections
            confluence_ok = self.confluence.test_connection()
            stash_ok = self.stash.test_connection()
            
            if not confluence_ok:
                logger.error("Failed to connect to Confluence")
            
            if not stash_ok:
                logger.error("Failed to connect to Stash/Bitbucket")
                
            self.initialized = confluence_ok and (stash_ok or True)  # Make Stash optional for now
            
            if self.initialized:
                logger.info("COPPER View-to-API Mapper initialized successfully")
                # Preload important pages
                self.mapper.load_mapping_knowledge()
            else:
                logger.error("COPPER View-to-API Mapper initialization failed")
                
        except Exception as e:
            logger.error(f"Error initializing COPPER View-to-API Mapper: {str(e)}")
            self.initialized = False
    
    def process_view_mapping(self, view_name: str) -> Dict[str, Any]:
        """
        Process mapping for a specific view.
        
        Args:
            view_name: Name of the view
            
        Returns:
            Dictionary with mapping results
        """
        logger.info(f"Processing mapping for view: {view_name}")
        
        try:
            # Step 1: Get the view definition from Stash
            sql_definition = self.stash.find_view_definition(view_name)
            
            if not sql_definition:
                logger.warning(f"Could not find SQL definition for view: {view_name}")
                return {
                    "success": False,
                    "error": f"Could not find SQL definition for view: {view_name}"
                }
                
            # Step 2: Parse the SQL definition
            view_info = ViewAnalyzer.parse_sql(sql_definition)
            
            if not view_info.get("success", False):
                logger.warning(f"Failed to parse SQL for view {view_name}: {view_info.get('error')}")
                return {
                    "success": False, 
                    "error": f"Failed to parse SQL: {view_info.get('error')}"
                }
                
            # Step 3: Map the view to API endpoints
            mapping_result = self.mapper.map_view_to_api(view_info)
            
            # Step 4: Generate mapping report
            mapping_report = self.mapper.generate_mapping_report(mapping_result)
            
            return {
                "success": True,
                "view_name": view_name,
                "sql_definition": sql_definition,
                "view_info": view_info,
                "mapping_result": mapping_result,
                "mapping_report": mapping_report
            }
            
        except Exception as e:
            logger.error(f"Error processing view mapping for {view_name}: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing view mapping: {str(e)}"
            }
    
    def process_sql_mapping(self, sql_query: str) -> Dict[str, Any]:
        """
        Process mapping for a SQL query string.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Dictionary with mapping results
        """
        logger.info("Processing mapping for SQL query")
        
        try:
            # Parse the SQL query
            view_info = ViewAnalyzer.parse_sql(sql_query)
            
            if not view_info.get("success", False):
                logger.warning(f"Failed to parse SQL query: {view_info.get('error')}")
                return {
                    "success": False, 
                    "error": f"Failed to parse SQL: {view_info.get('error')}"
                }
                
            # Map the view to API endpoints
            mapping_result = self.mapper.map_view_to_api(view_info)
            
            # Generate mapping report
            mapping_report = self.mapper.generate_mapping_report(mapping_result)
            
            return {
                "success": True,
                "view_name": view_info.get("view_name", "Custom Query"),
                "sql_definition": sql_query,
                "view_info": view_info,
                "mapping_result": mapping_result,
                "mapping_report": mapping_report
            }
            
        except Exception as e:
            logger.error(f"Error processing SQL mapping: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing SQL mapping: {str(e)}"
            }
    
    def get_context_for_question(self, question: str) -> Dict[str, Any]:
        """
        Get relevant context for a question.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with context information
        """
        logger.info(f"Getting context for question: {question}")
        
        context = {}
        
        # Try to identify view names in the question
        view_pattern = r'(?:view|table)\s+["\']?([A-Za-z0-9_]+)["\']?'
        view_matches = re.findall(view_pattern, question, re.IGNORECASE)
        
        # Try to identify API endpoints in the question
        api_pattern = r'(?:api|endpoint)\s+["\']?([A-Za-z0-9_/]+)["\']?'
        api_matches = re.findall(api_pattern, question, re.IGNORECASE)
        
        # Check for key phrases
        is_mapping_question = any(term in question.lower() for term in ["map", "mapping", "correspond", "relate", "equivalent"])
        is_faq_question = any(term in question.lower() for term in ["faq", "frequently", "asked", "question", "common", "how do i", "what is", "where can i"])
        is_api_doc_question = any(term in question.lower() for term in ["api doc", "documentation", "swagger", "reference", "endpoint", "parameter"])
        
        # If view name found, try to get view information
        if view_matches and is_mapping_question:
            view_name = view_matches[0]
            logger.info(f"Processing mapping for view mentioned in question: {view_name}")
            mapping_result = self.process_view_mapping(view_name)
            if mapping_result.get("success", False):
                context["mapping_result"] = mapping_result["mapping_result"]
                context["mapping_report"] = mapping_result["mapping_report"]
                context["view_info"] = mapping_result["view_info"]
        
        # Search Confluence for relevant content
        search_terms = question
        if view_matches:
            search_terms += f" view {view_matches[0]}"
        if api_matches:
            search_terms += f" api {api_matches[0]}"
        if is_mapping_question:
            search_terms += " mapping"
            
        search_results = self.confluence.search_content(search_terms, ConfigManager.CONFLUENCE_SPACES)
        if search_results:
            context["search_results"] = search_results
            
        # Add FAQ context if it's a FAQ-type question
        if is_faq_question and "faq" in self.mapper.important_pages:
            faq_page = self.mapper.important_pages["faq"]
            if faq_page:
                context["faq"] = faq_page["formatted_content"]
                
        # Add API documentation context if it's an API doc question
        if is_api_doc_question and "endpoints" in self.mapper.important_pages:
            api_doc_page = self.mapper.important_pages["endpoints"]
            if api_doc_page:
                context["api_docs"] = api_doc_page["formatted_content"]
        
        return context
    
    def answer_question(self, question: str) -> str:
        """
        Answer a user question.
        
        Args:
            question: User question
            
        Returns:
            Answer text
        """
        logger.info(f"Answering question: {question}")
        
        if not self.initialized:
            return "COPPER View-to-API Mapper is not properly initialized. Please check the configuration and logs."
            
        if not self.assistant.initialized():
            return "Gemini AI assistant is not available. Please check the configuration."
            
        # First check if this is a SQL query
        if "SELECT" in question.upper() and "FROM" in question.upper():
            # This might be a SQL query
            logger.info("Question appears to contain a SQL query, processing as SQL mapping")
            query_lines = [line for line in question.split('\n') if line.strip()]
            sql_query = "\n".join(query_lines)
            
            mapping_result = self.process_sql_mapping(sql_query)
            if mapping_result.get("success", False):
                context = {
                    "mapping_result": mapping_result["mapping_result"],
                    "mapping_report": mapping_result["mapping_report"],
                    "view_info": mapping_result["view_info"]
                }
                return self.assistant.answer_question(
                    "Please explain the mapping between this SQL query and the COPPER API endpoints.",
                    context
                )
        
        # Get relevant context for the question
        context = self.get_context_for_question(question)
        
        # Generate answer
        return self.assistant.answer_question(question, context)

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='COPPER View-to-API Mapper for CME BLR Hackathon 2025'
    )
    parser.add_argument(
        '--view', '-v',
        help='Process mapping for a specific view name'
    )
    parser.add_argument(
        '--sql', '-s',
        help='Process mapping for a SQL query file'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start in interactive mode'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run system tests'
    )
    args = parser.parse_args()
    
    # Initialize the mapper
    print("Initializing COPPER View-to-API Mapper...")
    mapper = CopperViewApiMapper()
    
    if not mapper.initialized:
        print("Initialization failed. Please check the logs for details.")
        return 1
        
    # Process specific view mapping
    if args.view:
        print(f"Processing mapping for view: {args.view}")
        result = mapper.process_view_mapping(args.view)
        if result.get("success", False):
            print(result["mapping_report"])
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        return 0
        
    # Process SQL query file
    if args.sql:
        try:
            with open(args.sql, 'r') as f:
                sql_query = f.read()
                
            print(f"Processing mapping for SQL query from file: {args.sql}")
            result = mapper.process_sql_mapping(sql_query)
            if result.get("success", False):
                print(result["mapping_report"])
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error reading SQL file: {str(e)}")
            
        return 0
        
    # Run system tests
    if args.test:
        print("Running system tests...")
        # Test Confluence connection
        print("Testing Confluence connection...")
        if mapper.confluence.test_connection():
            print("✅ Confluence connection successful")
        else:
            print("❌ Confluence connection failed")
            
        # Test Stash connection
        print("Testing Stash connection...")
        if mapper.stash.test_connection():
            print("✅ Stash connection successful")
        else:
            print("❌ Stash connection failed")
            
        # Test Gemini AI
        print("Testing Gemini AI...")
        if mapper.assistant.initialized():
            response = mapper.assistant.generate_response("Hello, are you working?")
            if response:
                print("✅ Gemini AI response received")
                print(f"Response: {response}")
            else:
                print("❌ Gemini AI response failed")
        else:
            print("❌ Gemini AI not initialized")
            
        return 0
        
    # Interactive mode (default if no other args provided)
    if args.interactive or not (args.view or args.sql or args.test):
        print("\n======== COPPER View-to-API Mapping Assistant ========")
        print(f"Initialized with {len(ConfigManager.CONFLUENCE_SPACES)} Confluence spaces.")
        print("I can help you understand how COPPER database views map to REST APIs.")
        print("What would you like to know about COPPER views or APIs?")
        
        while True:
            try:
                user_input = input("\nQuestion (type 'exit' to quit): ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Thanks for using the COPPER View-to-API Mapper. Have a great day!")
                    break
                    
                if not user_input:
                    continue
                    
                print("\nWorking on that for you...")
                start_time = time.time()
                answer = mapper.answer_question(user_input)
                end_time = time.time()
                
                print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
                print("-------")
                print(answer)
                print("-------")
                
            except KeyboardInterrupt:
                print("\nGoodbye! Feel free to come back if you have more questions.")
                break
                
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")
                print(f"Sorry, I ran into an issue: {str(e)}. Let's try a different question.")
                
    return 0

if __name__ == "__main__":
    sys.exit(main())
