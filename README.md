#!/usr/bin/env python3
"""
Enhanced COPPER Documentation Assistant
---------------------------------------
This comprehensive tool scans ALL content from a Confluence space,
with advanced extraction of tables, images, and structured data.
It uses intelligent relevance scoring without arbitrary limits
and provides direct source attribution in responses.
"""

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
import base64
from io import BytesIO
from urllib.parse import urlparse, urljoin
import hashlib

# Confluence imports
import requests
from bs4 import BeautifulSoup, NavigableString

# Gemini/Vertex AI imports
from vertexai.generative_models import GenerationConfig, GenerativeModel
import vertexai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("copper_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CopperAssistant")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://your-confluence-instance.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "xyz")  # Target specific space

# Performance and caching settings
MAX_WORKERS = 10  # Number of parallel workers for content fetching
CACHE_SIZE = 256  # Size of LRU cache for API responses
PAGE_CACHE_FILE = f"cache_{CONFLUENCE_SPACE}_pages.json"
CONTENT_CACHE_FILE = f"cache_{CONFLUENCE_SPACE}_content.json"

# Relevance settings
MIN_SCORE_THRESHOLD = 0.1  # Minimum relevance score to include content


class AdvancedContentExtractor:
    """Extract and process content from Confluence HTML, with focus on tables and images."""
    
    @staticmethod
    def get_text_with_context(elem):
        """Extract text from an element while preserving some context and formatting."""
        if isinstance(elem, NavigableString):
            return elem.strip()
        
        # For lists, preserve structure
        if elem.name in ['ul', 'ol']:
            items = []
            for li in elem.find_all('li', recursive=False):
                items.append(f"• {li.get_text(strip=True)}")
            return "\n".join(items)
        
        # For spans with styling
        if elem.name == 'span':
            if 'style' in elem.attrs:
                style = elem['style']
                text = elem.get_text(strip=True)
                if 'bold' in style or 'weight' in style:
                    return f"**{text}**"
                if 'italic' in style:
                    return f"*{text}*"
            return elem.get_text(strip=True)
        
        # For links
        if elem.name == 'a':
            text = elem.get_text(strip=True)
            href = elem.get('href', '')
            if href and not href.startswith('#'):
                return f"{text} [{href}]"
            return text
        
        # For other elements, just get text
        return elem.get_text(strip=True)
    
    @staticmethod
    def extract_table_to_markdown(table, include_caption=True):
        """
        Extract a table to a well-formatted markdown table.
        Handles complex tables with rowspans and colspans.
        """
        if not table:
            return ""
        
        # Get table caption if available
        caption = ""
        caption_elem = table.find('caption')
        if caption_elem and include_caption:
            caption = f"**Table: {caption_elem.get_text(strip=True)}**\n\n"
        
        # Find all rows
        rows = []
        thead = table.find('thead')
        tbody = table.find('tbody')
        tfoot = table.find('tfoot')
        
        # Process header
        header_row = []
        if thead:
            th_row = thead.find('tr')
            if th_row:
                header_row = [th.get_text(strip=True) for th in th_row.find_all(['th', 'td'])]
        
        # If no thead, try to use first row as header if it contains th elements
        if not header_row and tbody:
            first_row = tbody.find('tr')
            if first_row and first_row.find('th'):
                header_row = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
                # Skip this row in body processing
                rows_to_process = tbody.find_all('tr')[1:]
            else:
                rows_to_process = tbody.find_all('tr')
        elif tbody:
            rows_to_process = tbody.find_all('tr')
        else:
            # If no tbody, process all table rows
            all_rows = table.find_all('tr')
            if header_row:  # If we got a header, skip first row
                rows_to_process = all_rows[1:]
            else:
                # Try to use first row as header if it has th or looks like a header
                first_row = all_rows[0] if all_rows else None
                if first_row and (first_row.find('th') or all('header' in ' '.join(cell.get('class', [])) for cell in first_row.find_all(['td']))):
                    header_row = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
                    rows_to_process = all_rows[1:]
                else:
                    rows_to_process = all_rows
        
        # Process body rows
        for tr in rows_to_process:
            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if any(cell.strip() for cell in row):  # Skip empty rows
                rows.append(row)
        
        # Process footer if exists
        footer_rows = []
        if tfoot:
            for tr in tfoot.find_all('tr'):
                row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if any(cell.strip() for cell in row):
                    footer_rows.append(row)
        
        # Calculate column count
        if header_row:
            col_count = len(header_row)
        elif rows:
            col_count = max(len(row) for row in rows)
        else:
            col_count = 0
        
        if col_count == 0:
            return caption + "Table appears to be empty."
        
        # Calculate optimal column widths
        col_widths = [0] * col_count
        
        # Check header
        if header_row:
            for i, cell in enumerate(header_row[:col_count]):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Check data rows
        for row in rows:
            for i, cell in enumerate(row[:col_count]):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Check footer
        for row in footer_rows:
            for i, cell in enumerate(row[:col_count]):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Limit column width to reasonable maximum
        col_widths = [min(w, 30) for w in col_widths]
        
        # Construct markdown table
        md_table = []
        
        # Add header
        if header_row:
            # Pad header row to match column count
            padded_header = header_row + [''] * (col_count - len(header_row))
            header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(padded_header[:col_count])) + " |"
            md_table.append(header_line)
            
            # Add separator
            separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
            md_table.append(separator)
        
        # Add data rows
        for row in rows:
            # Pad row to match column count
            padded_row = row + [''] * (col_count - len(row))
            row_line = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_row[:col_count])) + " |"
            md_table.append(row_line)
        
        # Add footer rows
        if footer_rows:
            # Add separator before footer
            if rows and not header_row:  # Only add separator if we didn't already add one and have body rows
                separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
                md_table.append(separator)
            
            for row in footer_rows:
                # Pad row to match column count
                padded_row = row + [''] * (col_count - len(row))
                row_line = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_row[:col_count])) + " |"
                md_table.append(row_line)
        
        return caption + "\n".join(md_table)
    
    @staticmethod
    def get_image_context(img, soup):
        """Extract comprehensive context for an image including caption, alt text, and surrounding content."""
        context_parts = []
        
        # Get basic image attributes
        alt_text = img.get('alt', '').strip()
        title = img.get('title', '').strip()
        if alt_text:
            context_parts.append(f"Alt text: {alt_text}")
        if title:
            context_parts.append(f"Title: {title}")
        
        # Look for figure caption
        fig_caption = None
        parent_fig = img.find_parent('figure')
        if parent_fig:
            caption_elem = parent_fig.find('figcaption')
            if caption_elem:
                fig_caption = caption_elem.get_text(strip=True)
                context_parts.append(f"Caption: {fig_caption}")
        
        # Look for adjacent content that might describe the image
        # Check previous element
        prev_elem = img.find_previous_sibling(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if prev_elem and len(prev_elem.get_text(strip=True)) < 300:
            prev_text = prev_elem.get_text(strip=True)
            if prev_text and not any(part in prev_text for part in context_parts):
                context_parts.append(f"Previous content: {prev_text}")
        
        # Check next element (might be explaining the image)
        next_elem = img.find_next_sibling(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if next_elem and len(next_elem.get_text(strip=True)) < 300:
            next_text = next_elem.get_text(strip=True)
            if next_text and not any(part in next_text for part in context_parts):
                context_parts.append(f"Related content: {next_text}")
        
        # Check parent paragraph/div for context
        parent = img.find_parent(['p', 'div'])
        if parent and parent.name != 'figure':
            parent_text = parent.get_text(strip=True)
            # Remove the alt and title text from parent text to avoid duplication
            if alt_text:
                parent_text = parent_text.replace(alt_text, '')
            if title:
                parent_text = parent_text.replace(title, '')
            parent_text = parent_text.strip()
            
            if parent_text and len(parent_text) < 300 and not any(part in parent_text for part in context_parts):
                context_parts.append(f"Surrounding content: {parent_text}")
        
        # Check for possible labels or annotations
        for label in img.find_all_next(['span', 'label'], limit=3):
            if label.parent == img.parent and label.get_text(strip=True):
                label_text = label.get_text(strip=True)
                if not any(part in label_text for part in context_parts):
                    context_parts.append(f"Label: {label_text}")
        
        # Look for image dimensions which might help understand the image
        width = img.get('width', '')
        height = img.get('height', '')
        if width and height:
            context_parts.append(f"Dimensions: {width}x{height}")
        
        # Combine all context
        return " | ".join(context_parts)
    
    @staticmethod
    def extract_code_block(elem):
        """Extract code blocks with language information when available."""
        # Check for language specification
        language = ""
        if 'class' in elem.attrs:
            classes = elem['class']
            for cls in classes:
                if cls.startswith('language-'):
                    language = cls.replace('language-', '')
                    break
        
        code_content = elem.get_text()
        if language:
            return f"```{language}\n{code_content}\n```"
        else:
            return f"```\n{code_content}\n```"
    
    @staticmethod
    def extract_content_from_html(html_content, title=""):
        """
        Comprehensive extraction of content from HTML, with special focus on
        tables, images, and structural elements.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Setup content containers
            extracted = {
                "headings": [],
                "paragraphs": [],
                "tables": [],
                "images": [],
                "code_blocks": [],
                "lists": [],
                "notes": [],
                "metadata": {"title": title}
            }
            
            # Process headings to maintain document structure
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_level = int(heading.name[1])
                heading_text = heading.get_text(strip=True)
                if heading_text:
                    extracted["headings"].append({
                        "level": heading_level,
                        "text": heading_text,
                        "id": heading.get('id', '')
                    })
            
            # Process paragraphs with context
            for p in soup.find_all('p'):
                if p.get_text(strip=True):
                    # Check if paragraph contains or is near an image
                    has_image = bool(p.find('img'))
                    
                    # Get the paragraph text
                    para_text = p.get_text(strip=True)
                    
                    # Only add if it has content
                    if para_text:
                        extracted["paragraphs"].append({
                            "text": para_text,
                            "has_image": has_image
                        })
            
            # Process tables with enhanced extraction
            for i, table in enumerate(soup.find_all('table')):
                table_md = AdvancedContentExtractor.extract_table_to_markdown(table)
                if table_md:
                    extracted["tables"].append({
                        "index": i,
                        "markdown": table_md,
                        "original": str(table)
                    })
            
            # Process images with comprehensive context
            for img in soup.find_all('img'):
                context = AdvancedContentExtractor.get_image_context(img, soup)
                src = img.get('src', '')
                alt = img.get('alt', '')
                
                extracted["images"].append({
                    "src": src,
                    "alt": alt,
                    "context": context
                })
            
            # Process code blocks
            for pre in soup.find_all('pre'):
                code = pre.find('code')
                if code:
                    code_block = AdvancedContentExtractor.extract_code_block(code)
                    extracted["code_blocks"].append(code_block)
                else:
                    # Pre without code tag
                    extracted["code_blocks"].append(f"```\n{pre.get_text(strip=True)}\n```")
            
            # Process lists
            for list_elem in soup.find_all(['ul', 'ol']):
                list_items = []
                for li in list_elem.find_all('li', recursive=False):
                    list_items.append(li.get_text(strip=True))
                
                if list_items:
                    list_type = "unordered" if list_elem.name == "ul" else "ordered"
                    extracted["lists"].append({
                        "type": list_type,
                        "items": list_items
                    })
            
            # Process notes, warnings, info panels
            for div in soup.find_all(['div', 'section']):
                if 'class' in div.attrs:
                    # Look for common Confluence structured content classes
                    class_str = ' '.join(div['class'])
                    if any(term in class_str for term in ['panel', 'info', 'note', 'warning', 'callout', 'aui-message']):
                        title_elem = div.find(['h3', 'h4', 'h5', 'strong', 'b'])
                        title = title_elem.get_text(strip=True) if title_elem else "Note"
                        content = div.get_text(strip=True)
                        note_type = "info"
                        if "warning" in class_str or "error" in class_str:
                            note_type = "warning"
                        elif "note" in class_str:
                            note_type = "note"
                        elif "success" in class_str:
                            note_type = "success"
                        
                        extracted["notes"].append({
                            "type": note_type,
                            "title": title,
                            "content": content
                        })
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting HTML content: {str(e)}")
            # Return minimal structure with error message
            return {
                "headings": [],
                "paragraphs": [{"text": f"Error extracting content: {str(e)}", "has_image": False}],
                "tables": [],
                "images": [],
                "code_blocks": [],
                "lists": [],
                "notes": [],
                "metadata": {"title": title, "error": str(e)}
            }
    
    @staticmethod
    def format_for_context(extracted_content, include_title=True):
        """
        Format the extracted content into a well-structured string for context.
        Focuses on preserving tables, images, and document structure.
        """
        parts = []
        
        # Add title if available and requested
        if include_title and extracted_content["metadata"].get("title"):
            parts.append(f"# {extracted_content['metadata']['title']}")
        
        # Combine headings and paragraphs to maintain document flow
        content_flow = []
        
        # Get headings with their position indicators
        headings_with_pos = [(i, h) for i, h in enumerate(extracted_content["headings"])]
        
        # Get paragraphs with position indicators (rough estimate as they come after headings)
        paragraph_pos = len(headings_with_pos)
        paragraphs_with_pos = [(paragraph_pos + i, {"type": "paragraph", "content": p}) 
                              for i, p in enumerate(extracted_content["paragraphs"])]
        
        # Combine and sort by position
        all_content = [(pos, {"type": "heading", "content": h}) for pos, h in headings_with_pos]
        all_content.extend(paragraphs_with_pos)
        all_content.sort(key=lambda x: x[0])
        
        # Process in order
        for _, item in all_content:
            if item["type"] == "heading":
                heading = item["content"]
                content_flow.append(f"{'#' * heading['level']} {heading['text']}")
            else:
                paragraph = item["content"]
                content_flow.append(paragraph["text"])
        
        if content_flow:
            parts.append("\n\n".join(content_flow))
        
        # Add tables with proper formatting
        if extracted_content["tables"]:
            tables_section = []
            for table in extracted_content["tables"]:
                tables_section.append(table["markdown"])
            parts.append("\n\n".join(tables_section))
        
        # Add code blocks
        if extracted_content["code_blocks"]:
            parts.append("\n\n".join(extracted_content["code_blocks"]))
        
        # Add list content
        if extracted_content["lists"]:
            lists_section = []
            for list_item in extracted_content["lists"]:
                if list_item["type"] == "unordered":
                    formatted_list = "\n".join([f"• {item}" for item in list_item["items"]])
                else:  # ordered
                    formatted_list = "\n".join([f"{i+1}. {item}" for i, item in enumerate(list_item["items"])])
                lists_section.append(formatted_list)
            parts.append("\n\n".join(lists_section))
        
        # Add notes and callouts
        if extracted_content["notes"]:
            notes_section = []
            for note in extracted_content["notes"]:
                notes_section.append(f"**{note['title']}**: {note['content']}")
            parts.append("\n\n".join(notes_section))
        
        # Add image descriptions with context
        if extracted_content["images"]:
            images_section = []
            for img in extracted_content["images"]:
                desc = f"[IMAGE: {img['alt'] if img['alt'] else 'Unnamed image'}"
                if img["context"]:
                    desc += f" | {img['context']}"
                desc += "]"
                images_section.append(desc)
            parts.append("\n\n".join(images_section))
        
        # Join all sections with double newlines
        return "\n\n".join(parts)


class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling and caching."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://mycompany.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.api_url = f"{self.base_url}/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "COPPER-AI-Python-Agent"
        }
        self.session = requests.Session()
        # Set up a default request timeout
        self.timeout = 30
        # Cache for API responses
        self.cache = {}
        # Thread lock for the cache
        self.cache_lock = threading.Lock()
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                timeout=self.timeout,
                verify=False  # Using verify=False as requested
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info("Connection to Confluence successful!")
                return True
            else:
                logger.warning(f"Empty response received during connection test")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _get_cached_request(self, url, params_str):
        """Cached version of GET requests to reduce API calls."""
        try:
            params = json.loads(params_str)
            response = self.session.get(
                url,
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
                verify=False  # Using verify=False as requested
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
        cache_path = PAGE_CACHE_FILE
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('space_key') == space_key:
                        logger.info(f"Using cached page list from {cache_path}")
                        return cached_data.get('pages', [])
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        # If no cache, fetch all pages
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
                response_text = self._get_cached_request(f"{self.api_url}/content", params_str)
                
                if not response_text:
                    logger.warning(f"Empty response when fetching pages at start={start}")
                    break
                
                response_data = json.loads(response_text)
                
                results = response_data.get("results", [])
                all_pages.extend(results)
                
                # Check if there are more pages
                if "size" in response_data and "limit" in response_data:
                    if response_data["size"] < response_data["limit"]:
                        has_more = False
                    else:
                        start += batch_size
                else:
                    has_more = False
                
                logger.info(f"Fetched {len(results)} pages, total so far: {len(all_pages)}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching pages: {str(e)}")
                break
        
        # Cache the results
        try:
            with open(cache_path, 'w') as f:
                json.dump({'space_key': space_key, 'pages': all_pages}, f)
                logger.info(f"Cached {len(all_pages)} pages to {cache_path}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
        
        logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
        return all_pages
    
    def get_page_content(self, page_id, expand=None):
        """
        Get the content of a page in a suitable format for NLP.
        Includes comprehensive extraction of tables, images, and document structure.
        
        Args:
            page_id: The ID of the page
        """
        try:
            # Calculate cache key based on page ID
            cache_key = f"page_content_{page_id}"
            
            # Load content cache from disk if exists
            content_cache_path = CONTENT_CACHE_FILE
            content_cache = {}
            
            if os.path.exists(content_cache_path):
                try:
                    with open(content_cache_path, 'r') as f:
                        content_cache = json.load(f)
                        if cache_key in content_cache:
                            return content_cache[cache_key]
                except Exception as e:
                    logger.warning(f"Error reading content cache file: {str(e)}")
            
            # If not in cache, fetch and process
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}",
                "space_key": page.get("space", {}).get("key", ""),
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process with our advanced content extractor
            extracted_content = AdvancedContentExtractor.extract_content_from_html(html_content, page.get("title", ""))
            formatted_content = AdvancedContentExtractor.format_for_context(extracted_content)
            
            result = {
                "metadata": metadata,
                "content": formatted_content,
                "extracted": extracted_content,
                "raw_html": html_content
            }
            
            # Cache the result
            try:
                content_cache[cache_key] = result
                with open(content_cache_path, 'w') as f:
                    json.dump(content_cache, f)
            except Exception as e:
                logger.warning(f"Error writing content to cache file: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma separated list of properties to expand (e.g., "body.storage,version,space")
        """
        try:
            # Use cached version if available
            cache_key = f"content_{content_id}_{expand}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            params = {}
            if expand:
                params["expand"] = expand
                
            # Convert params to string for cache key
            params_str = json.dumps(params, sort_keys=True)
            
            # Try to get from cache first
            response_text = self._get_cached_request(f"{self.api_url}/content/{content_id}", params_str)
            
            if not response_text:
                logger.warning(f"Empty response received when retrieving content ID: {content_id}")
                return None
            
            content = json.loads(response_text)
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = content
            
            return content
                
        except Exception as e:
            logger.error(f"Error getting content by ID {content_id}: {str(e)}")
            return None


class GeminiAssistant:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")
        
    def generate_response(self, prompt, context=None, include_sources=True):
        """
        Generate a response from Gemini based on the prompt and context.
        
        Args:
            prompt: The user's question or prompt
            context: Context information from Confluence
            include_sources: Whether to include source URLs in the response
            
        Returns:
            The generated response
        """
        logger.info(f"Generating response for prompt: {prompt}")
        
        try:
            # Create a system prompt that instructs Gemini on how to use the context
            system_prompt = """
            You are the friendly COPPER Assistant, an expert on mapping database views to REST APIs.
            
            Your personality:
            - Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
            - Explain technical concepts in plain language, as if speaking to a colleague
            - Use simple analogies and examples to clarify complex ideas
            - Be concise but thorough - focus on answering the question directly first, then add helpful context
            
            Your expertise:
            - Deep knowledge of the COPPER database system, its views, and corresponding API endpoints
            - Understanding database-to-API mapping patterns and best practices
            - Awareness of how applications integrate with COPPER's REST APIs
            - Expert in interpreting table structures, field mappings, and API parameters
            
            When answering:
            1. Always include SPECIFIC information when available (like the FULL FORM of acronyms, complete API endpoint lists, detailed table data)
            2. Answer the user's question DIRECTLY first, then provide additional context
            3. Format tables and structured data clearly to enhance readability
            4. Use bullet points for lists and steps
            5. INCLUDE THE SOURCE URLS at the end of your response, citing which pages contained the information you used
            6. If data appears in an image or table, SPECIFICALLY mention this: "According to the table in [URL]..." or "As shown in the diagram at [URL]..."
            
            You must be COMPREHENSIVE and THOROUGH. If asked about things like "the full form of COPPER" or "all API endpoints", make sure your answer includes ALL the relevant information from the documentation.
            """
            
            # Craft the full prompt with context
            full_prompt = system_prompt + "\n\n"
            
            if context:
                # Prepare context sections
                sources = []
                context_text = ""
                
                if isinstance(context, dict):
                    # Format from dict structure
                    for source_url, content in context.items():
                        if content.strip():
                            context_text += f"\n\n--- BEGIN CONTENT FROM: {source_url} ---\n\n"
                            context_text += content
                            context_text += f"\n\n--- END CONTENT FROM: {source_url} ---\n\n"
                            sources.append(source_url)
                elif isinstance(context, str):
                    # If it's just a string, use it directly
                    context_text = context
                
                # Trim context if it's too large
                if len(context_text) > 30000:  # Leave room for system prompt and response
                    logger.warning(f"Context too large ({len(context_text)} chars), trimming...")
                    # Try to trim at document boundaries
                    sections = re.split(r'---\s*BEGIN CONTENT FROM:', context_text)
                    
                    if len(sections) > 1:
                        # Keep first part (intro) and add sections until limit
                        trimmed_context = sections[0]
                        for section in sections[1:]:
                            if len(trimmed_context) + len(section) + 30 < 30000:
                                trimmed_context += "--- BEGIN CONTENT FROM:" + section
                            else:
                                break
                        context_text = trimmed_context
                    else:
                        # If splitting didn't work, just trim to size
                        context_text = context_text[:30000]
                    
                    logger.info(f"Trimmed context to {len(context_text)} chars")
                
                full_prompt += "CONTEXT INFORMATION:\n" + context_text + "\n\n"
                
            full_prompt += f"USER QUESTION: {prompt}\n\nResponse:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.95,
                max_output_tokens=2048,
            )
            
            # Generate the response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            if response.candidates and response.candidates[0].text:
                response_text = response.candidates[0].text.strip()
                logger.info(f"Successfully generated response ({len(response_text)} chars)")
                
                # If sources should be included but aren't already
                if include_sources and isinstance(context, dict) and sources:
                    if not any(url in response_text for url in sources):
                        response_text += "\n\nSources:\n"
                        for url in sources:
                            response_text += f"- {url}\n"
                
                return response_text
            else:
                logger.warning("No response generated from Gemini")
                return "I couldn't find a specific answer to that question in our documentation. Could you try rephrasing, or maybe I can help you find the right documentation to look at?"
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I ran into a technical issue while looking that up. Let me know if you'd like to try a different question or approach."


class CopperAssistant:
    """Main class that coordinates between Confluence and Gemini."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None):
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.gemini = GeminiAssistant()
        self.space_pages = []  # All pages in the space
        self.space_key = space_key
        self.page_content_cache = {}  # Cache for page content to avoid re-fetching
        logger.info(f"Initialized Copper Assistant targeting space: {space_key or 'all spaces'}")
        
    def initialize(self):
        """Initialize by testing connections and gathering initial space content."""
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and URL.")
            return False
            
        logger.info("Loading space content...")
        self.load_space_content()
        return True
        
    def load_space_content(self):
        """Load metadata for all pages in the specified space."""
        if not self.space_key:
            logger.error("No space key specified. Please provide a space key.")
            return
        
        # Get all pages in the space
        self.space_pages = self.confluence.get_all_pages_in_space(self.space_key)
        
        logger.info(f"Loaded metadata for {len(self.space_pages)} pages from space {self.space_key}")
    
    def _fetch_page_content(self, page_id, page_title):
        """Helper to fetch page content with error handling."""
        try:
            # Check cache first
            if page_id in self.page_content_cache:
                return self.page_content_cache[page_id]
            
            page_content = self.confluence.get_page_content(page_id)
            if page_content:
                # Cache the content
                self.page_content_cache[page_id] = page_content
                return page_content
            else:
                logger.warning(f"Failed to get content for page {page_title} ({page_id})")
                return None
        except Exception as e:
            logger.error(f"Error fetching content for page {page_title} ({page_id}): {str(e)}")
            return None
    
    def _fetch_page_content_batch(self, pages):
        """Fetch content for a batch of pages in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(self._fetch_page_content, page["id"], page["title"]): page["id"]
                for page in pages
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_id = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        results[page_id] = content
                except Exception as e:
                    logger.error(f"Error in page content fetching task for {page_id}: {str(e)}")
        
        return results
    
    def search_all_pages(self, query):
        """
        Search all pages with a comprehensive scoring system designed to find
        ALL relevant content, including implicit matches and domain-specific relevance.
        """
        if not self.space_pages:
            return []
        
        # Extract query components
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Define domain-specific terms and weights
        domain_terms = {
            # Core COPPER terms
            "copper": 10.0, "api": 8.0, "endpoint": 8.0, "database": 8.0, "view": 8.0, 
            "schema": 7.0, "table": 7.0, "column": 7.0, "field": 7.0, "attribute": 7.0,
            
            # Relationship terms
            "mapping": 9.0, "relation": 6.0, "join": 5.0, "foreign key": 6.0, "primary key": 6.0,
            
            # API terms
            "rest": 7.0, "request": 6.0, "response": 6.0, "parameter": 6.0, "json": 5.0,
            "get": 5.0, "post": 5.0, "put": 5.0, "delete": 5.0, "header": 5.0,
            
            # Documentation terms
            "document": 3.0, "manual": 3.0, "guide": 3.0, "reference": 3.0, "example": 4.0,
            
            # Question-specific terms
            "full form": 15.0, "meaning": 10.0, "acronym": 10.0, "definition": 10.0,
            "all endpoints": 15.0, "list of": 8.0, "available": 5.0, "complete": 8.0
        }
        
        # Handle special query cases
        special_query_terms = {
            "full form": ["meaning", "acronym", "definition", "stand for", "stands for", "full form"],
            "api endpoints": ["endpoints", "api", "rest", "services", "routes", "controller"],
            "database structure": ["schema", "table", "column", "view", "structure", "database"],
            "mapping": ["mapping", "relation", "connection", "between", "link", "corresponds"]
        }
        
        # Add special case search terms if applicable
        for special_case, terms in special_query_terms.items():
            if any(term in query_lower for term in terms):
                query_words.update(terms)
        
        # Score all pages
        page_scores = []
        for page in self.space_pages:
            page_id = page["id"]
            title = page.get("title", "").lower()
            
            # Initial score based on title
            score = 0
            
            # Title relevance
            for word in query_words:
                if word in title:
                    # Words in title are heavily weighted
                    weight = domain_terms.get(word, 1.0)
                    score += weight * 5
            
            # Special title bonuses
            for special_term in ["copper", "api", "database", "glossary", "guide", "reference"]:
                if special_term in title:
                    score += domain_terms.get(special_term, 1.0) * 2
            
            # For common information-seeking questions about copper itself
            if ("what" in query_lower or "definition" in query_lower or "mean" in query_lower) and "copper" in query_lower:
                if "about" in title.lower() or "overview" in title.lower() or "introduction" in title.lower():
                    score += 15
            
            # For api endpoint listing
            if ("endpoint" in query_lower or "all" in query_lower or "list" in query_lower) and "api" in query_lower:
                if "endpoint" in title.lower() or "api" in title.lower() or "reference" in title.lower():
                    score += 20
            
            # Record the score
            page_scores.append((page, score))
        
        # Sort by relevance score
        page_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates (more inclusive to avoid missing relevant pages)
        # Instead of a fixed limit, use a score threshold relative to the top score
        if page_scores:
            top_score = page_scores[0][1]
            score_threshold = max(MIN_SCORE_THRESHOLD, top_score * 0.2)  # At least 20% of top score
            candidates = [p for p, s in page_scores if s >= score_threshold]
            
            # If we have very few candidates, take at least 15-20 pages to be safe
            if len(candidates) < 15:
                candidates = [p for p, s in page_scores[:20] if s > 0]
        else:
            candidates = []
        
        logger.info(f"Selected {len(candidates)} candidate pages for search: {query}")
        return candidates
    
    def extract_relevant_content(self, query):
        """
        Comprehensive content extraction that prioritizes finding ALL relevant information,
        with special handling of tables and images.
        """
        # First get candidate pages
        candidate_pages = self.search_all_pages(query)
        
        if not candidate_pages:
            return "I couldn't find any relevant information in the Confluence space."
        
        # Fetch content for candidates
        page_contents = self._fetch_page_content_batch(candidate_pages)
        
        # Generate focused context
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Create context with source URLs as keys
        relevant_content = {}
        
        # Special handling for different query types
        is_definition_query = any(term in query_lower for term in ["meaning", "definition", "full form", "acronym", "stand for", "what is"])
        is_listing_query = any(term in query_lower for term in ["all", "list", "every", "complete", "available"])
        is_table_query = any(term in query_lower for term in ["table", "column", "field", "schema", "structure"])
        is_image_query = any(term in query_lower for term in ["diagram", "picture", "image", "screenshot", "workflow"])
        
        # Track scores to dynamically adjust threshold
        content_scores = []
        
        # Process each page
        for page_id, content in page_contents.items():
            if not content:
                continue
                
            source_url = content["metadata"]["url"]
            page_title = content["metadata"]["title"]
            page_content = content["content"]
            
            # Calculate relevance score for this content
            score = 0
            
            # Check for table content if this is a table-related query
            has_tables = False
            if "TABLE:" in page_content:
                has_tables = True
                if is_table_query:
                    score += 15
            
            # Check for image content if this is an image-related query
            has_images = False
            if "[IMAGE:" in page_content:
                has_images = True
                if is_image_query:
                    score += 10
            
            # For definition queries, prioritize content with the term near "means", "is", etc.
            if is_definition_query:
                # Look for definition patterns
                matches = []
                for term in query_words:
                    if term in ["what", "is", "the", "of", "a", "an", "means", "meaning"]:
                        continue
                    
                    patterns = [
                        rf"(?i){term}\s+(?:stands for|means|is short for|is an acronym for|is)",
                        rf"(?i)(?:meaning of|definition of|full form of)\s+{term}",
                        rf"(?i){term}.*(?:acronym|abbreviation)"
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, page_content):
                            matches.append(term)
                            score += 25  # High score for definition matches
                
                # If we found definition matches, boost score significantly
                if matches:
                    logger.info(f"Found definition matches for {matches} in {page_title}")
                    score += 50
            
            # For listing queries, prioritize pages with lists, tables or enumeration
            if is_listing_query:
                # Look for list indicators
                if "• " in page_content or any(f"{i}. " in page_content for i in range(1, 10)):
                    score += 15
                
                # Look for "available", "supported", "all" near key terms
                for term in query_words:
                    if term in ["all", "list", "every", "available"]:
                        continue
                    
                    patterns = [
                        rf"(?i)all\s+(?:\w+\s+)*{term}",
                        rf"(?i)available\s+(?:\w+\s+)*{term}",
                        rf"(?i){term}.*(?:supported|available|provided)"
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, page_content):
                            score += 20
            
            # Score based on query term frequency and position
            for word in query_words:
                word_freq = page_content.lower().count(word)
                if word_freq > 0:
                    # Basic frequency score
                    score += min(word_freq, 10) * 0.5
                    
                    # Check if word appears in important locations
                    lower_content = page_content.lower()
                    sections = lower_content.split("\n\n")
                    
                    for i, section in enumerate(sections):
                        if word in section:
                            # Words in early sections get higher weight
                            position_weight = max(1.0, 3.0 - (i * 0.1))
                            score += position_weight
                            
                            # Check for word in headings (higher weight)
                            if re.search(rf"(?i)^#+.*{word}.*$", section, re.MULTILINE):
                                score += 5
            
            # Extra score for tables/images based on query
            if has_tables and any(term in query_lower for term in ["table", "data", "field", "column"]):
                score += 10
            
            if has_images and any(term in query_lower for term in ["image", "diagram", "picture"]):
                score += 10
            
            content_scores.append((source_url, page_title, page_content, score))
        
        # Sort by score
        content_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Determine score threshold dynamically
        if content_scores:
            top_score = content_scores[0][3]
            # More relaxed threshold to include potentially relevant content
            score_threshold = max(0.1, top_score * 0.15)  # At least 15% of top score
            
            # Log top scoring pages
            logger.info(f"Top scoring pages for query '{query}':")
            for url, title, _, score in content_scores[:5]:
                logger.info(f"  - {title}: {score:.2f}")
            
            # Add content that meets threshold to the context
            for source_url, page_title, page_content, score in content_scores:
                if score >= score_threshold:
                    relevant_content[source_url] = page_content
                    logger.info(f"Including content from {page_title} (score: {score:.2f})")
        
        if not relevant_content:
            return "I couldn't find specific information related to your question in the Confluence space."
        
        return relevant_content
    
    def answer_question(self, question):
        """Answer a question using Confluence content and Gemini."""
        logger.info(f"Processing question: {question}")
        
        # Extract relevant content based on the question
        relevant_content = self.extract_relevant_content(question)
        
        # Generate response using Gemini
        response = self.gemini.generate_response(question, relevant_content, include_sources=True)
        
        return response


def main():
    """Main entry point for the COPPER Assistant."""
    logger.info("Starting COPPER Assistant")
    
    # Check for required environment variables
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN or not CONFLUENCE_URL:
        logger.error("Missing Confluence credentials. Please set CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, and CONFLUENCE_URL environment variables.")
        print("Error: Missing Confluence credentials. Please set the required environment variables.")
        return
        
    print("\nInitializing COPPER Assistant...")
    print("Connecting to Confluence and loading knowledge base...")
    
    # Initialize the assistant
    assistant = CopperAssistant(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, space_key=CONFLUENCE_SPACE)
    
    if not assistant.initialize():
        logger.error("Failed to initialize COPPER Assistant.")
        print("Error: Failed to initialize. Please check the logs for details.")
        return
        
    print(f"\n===== COPPER Database-to-API Mapping Assistant =====")
    print(f"I've loaded information from all pages in the {CONFLUENCE_SPACE} space.")
    print("I can help you understand the COPPER database structure, APIs, mappings, and more.")
    print("What would you like to know about COPPER?")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Thanks for using the COPPER Assistant. Have a great day!")
                break
                
            if not user_input:
                continue
                
            print("\nSearching for information...")
            start_time = time.time()
            answer = assistant.answer_question(user_input)
            end_time = time.time()
            
            print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
            print("-------")
            print(answer)
            print("-------")
            
        except KeyboardInterrupt:
            print("\nGoodbye! Feel free to come back if you have more questions.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Sorry, I ran into an issue: {str(e)}. Let's try a different question.")


if __name__ == "__main__":
    main()
