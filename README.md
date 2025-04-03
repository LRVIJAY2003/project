#!/usr/bin/env python3
"""
COPPER View to API Mapper - Enhanced Complete Space Scanner
----------------------------------------------------------
This optimized tool fetches ALL content from a Confluence space,
efficiently extracts structured data including tables and image contextual information,
and processes it with Gemini AI to provide conversational, expert responses about
database views and REST API mappings.
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
import queue
import threading

# Confluence imports
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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
PROJECT_ID = os.environ.get("PROJECT_ID", )
REGION = os.environ.get("REGION", )
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL",)
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "xyz")  # Target specific space

# Performance settings
MAX_WORKERS = 5  # Number of parallel workers for content fetching
CACHE_SIZE = 128  # Size of LRU cache for API responses
PAGE_CACHE_FILE = f"cache_{CONFLUENCE_SPACE}_pages.json"


class ContentExtractor:
    """Extract and process content from Confluence HTML, including tables and images."""
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def extract_content_from_html(html_content, title=""):
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
                table_title = caption.text.strip() if caption else f"Table {i+1}"
                
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
                        # Check if it looks like a header row (has th elements or all cells look like headers)
                        first_row_cells = first_row.find_all(['th', 'td'])
                        if first_row.find('th') or all(cell.name == 'td' and cell.get('class') and 'header' in ' '.join(cell.get('class', [])) for cell in first_row_cells):
                            headers = [th.text.strip() for th in first_row_cells]
                
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
                table_text = [f"TABLE: {table_title}"]
                
                # Format with consistent column widths for better readability
                if headers and rows:
                    # Calculate column widths
                    col_widths = [len(h) for h in headers]
                    for row in rows:
                        for i, cell in enumerate(row[:len(col_widths)]):
                            col_widths[i] = max(col_widths[i], len(cell))
                    
                    # Format header row
                    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
                    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
                    table_text.append(header_row)
                    table_text.append(separator)
                    
                    # Format data rows
                    for row in rows:
                        # Pad row if needed to match header length
                        if len(row) < len(headers):
                            row.extend([""] * (len(headers) - len(row)))
                        # Truncate if longer than headers
                        row = row[:len(headers)]
                        table_text.append("| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
                
                elif rows:  # Table with no headers
                    # Calculate column widths
                    max_cols = max(len(row) for row in rows)
                    col_widths = [0] * max_cols
                    for row in rows:
                        for i, cell in enumerate(row[:max_cols]):
                            col_widths[i] = max(col_widths[i], len(cell))
                    
                    # Format rows
                    for row in rows:
                        # Pad row if needed to match max columns
                        if len(row) < max_cols:
                            row.extend([""] * (max_cols - len(row)))
                        table_text.append("| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
                
                if len(table_text) > 1:  # Only add tables with content
                    tables.append("\n".join(table_text))
            
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
                    prev_elem = img.find_previous_sibling(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if prev_elem and len(prev_elem.text.strip()) < 200:  # Only short contexts
                        context = f"Previous content: {prev_elem.text.strip()}"
                
                # Construct meaningful image description
                desc = alt_text or title or "Image"
                if context:
                    desc += f" - {context}"
                
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
                        code_blocks.append(f"```{lang}\n{code_content}\n```")
                    else:
                        code_blocks.append(f"```\n{code_content}\n```")
                else:
                    # Pre without code tag
                    code_blocks.append(f"```\n{pre.text.strip()}\n```")
            
            # Extract any important structured content
            structured_content = []
            for div in soup.find_all(['div', 'section']):
                if 'class' in div.attrs:
                    # Look for common Confluence structured content classes
                    class_str = ' '.join(div['class'])
                    if any(term in class_str for term in ['panel', 'info', 'note', 'warning', 'callout', 'aui-message']):
                        title_elem = div.find(['h3', 'h4', 'h5', 'strong', 'b'])
                        title = title_elem.text.strip() if title_elem else "Note"
                        content = div.text.strip()
                        structured_content.append(f"--- {title} ---\n{content}")
            
            return {
                "text": "\n\n".join(text_content),
                "tables": tables,
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
            sections.append(f"## {title}")
        
        if extracted_content["text"]:
            sections.append(extracted_content["text"])
        
        if extracted_content["tables"]:
            for table in extracted_content["tables"]:
                sections.append(f"\n{table}")
        
        if extracted_content["code_blocks"]:
            sections.append("\n\nCode Examples:")
            sections.extend(extracted_content["code_blocks"])
        
        if extracted_content["structured_content"]:
            sections.append("\n\nImportant Notes:")
            sections.extend(extracted_content["structured_content"])
        
        if extracted_content["images"]:
            sections.append("\n\nImage Information:")
            sections.extend(extracted_content["images"])
        
        return "\n\n".join(sections)


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
                time.sleep(0.2)
                
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
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        """
        try:
            # Use cached version if available
            cache_key = f"page_content_{page_id}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process with our advanced content extractor
            extracted_content = ContentExtractor.extract_content_from_html(html_content, page.get("title", ""))
            formatted_content = ContentExtractor.format_for_context(extracted_content, page.get("title", ""))
            
            result = {
                "metadata": metadata,
                "content": formatted_content,
                "raw_html": html_content
            }
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = result
            
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
        
    def generate_response(self, prompt, copper_context=None):
        """
        Generate a response from Gemini based on the prompt and COPPER context.
        
        Args:
            prompt: The user's question or prompt
            copper_context: Context information about COPPER (from Confluence)
        
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
            - Add occasional light humor where appropriate to make the conversation engaging
            - Be concise but thorough - focus on answering the question directly first, then add helpful context
            
            Your expertise:
            - Deep knowledge of the COPPER database system, its views, and corresponding API endpoints
            - Understanding database-to-API mapping patterns and best practices
            - Awareness of how applications integrate with COPPER's REST APIs
            - Expert in interpreting table structures, field mappings, and API parameters
            
            When answering:
            1. Directly address the user's question first
            2. Provide practical, actionable information when possible
            3. Format tables and structured data clearly to enhance readability
            4. Use bullet points or numbered lists for steps or multiple items
            5. Reference specific examples from the documentation when available
            6. Acknowledge any limitations in the available information
            
            Remember to maintain a balance between being friendly and professional - you're a helpful colleague, not a formal technical document.
            """
            
            # Craft the full prompt with context
            full_prompt = system_prompt + "\n\n"
            
            if copper_context:
                # Trim context if it's too large
                if len(copper_context) > 28000:  # Leave room for system prompt and response
                    logger.warning(f"Context too large ({len(copper_context)} chars), trimming...")
                    # Try to trim at paragraph boundaries
                    paragraphs = copper_context.split("\n\n")
                    trimmed_context = ""
                    for para in paragraphs:
                        if len(trimmed_context) + len(para) + 2 < 28000:
                            trimmed_context += para + "\n\n"
                        else:
                            break
                    copper_context = trimmed_context
                    logger.info(f"Trimmed context to {len(copper_context)} chars")
                
                full_prompt += "CONTEXT INFORMATION:\n" + copper_context + "\n\n"
                
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
    
    def extract_relevant_content(self, query):
        """
        Extract content from pages that is most relevant to the query.
        Uses an efficient search algorithm to find the most relevant pages.
        """
        if not self.space_pages:
            return "No pages found in the specified Confluence space."
            
        # Step 1: Score pages based on title and perform initial filtering
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Add some domain-specific terms if query is very short
        domain_terms = {
            "copper", "database", "view", "api", "endpoint", "rest", "mapping", 
            "column", "table", "schema", "field", "attribute"
        }
        
        if len(query_words) < 3:
            query_words.update(domain_terms)
        
        # Initial scoring based on title to filter down to potential candidates
        candidates = []
        for page in self.space_pages:
            title = page.get("title", "").lower()
            # Quick score based on title matches
            title_score = sum(1 for word in query_words if word in title)
            
            # If title has matches, or it mentions key terms, include as candidate
            if title_score > 0 or any(term in title for term in domain_terms):
                candidates.append((page, title_score))
        
        # Sort candidates by initial score and take top 20 for detailed analysis
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:20]
        
        logger.info(f"Selected {len(top_candidates)} candidate pages for detailed analysis")
        
        # Fetch content for candidate pages in parallel
        page_dict = {p[0]["id"]: p[0] for p in top_candidates}
        candidate_pages = list(page_dict.values())
        
        if not candidate_pages:
            # If no candidates found, take 10 most recently updated pages
            recent_pages = sorted(self.space_pages, 
                                 key=lambda p: p.get("history", {}).get("lastUpdated", "2000-01-01"),
                                 reverse=True)[:10]
            candidate_pages = recent_pages
            page_dict = {p["id"]: p for p in recent_pages}
        
        # Fetch content for candidates
        page_contents = self._fetch_page_content_batch(candidate_pages)
        
        # Step 2: Detailed relevance scoring with content
        scored_pages = []
        for page_id, content in page_contents.items():
            page = page_dict[page_id]
            page_text = content["content"].lower()
            title = page["title"].lower()
            
            # Calculate a relevance score
            score = 0
            
            # Score based on query word frequency
            for word in query_words:
                word_count = page_text.count(word)
                score += word_count * 0.1  # Base score per occurrence
                
                # Higher score for words in title
                if word in title:
                    score += 5
            
            # Bonus for exact phrase matches
            query_lower = query.lower()
            if query_lower in page_text:
                score += 50  # Huge bonus for exact match
            else:
                # Look for phrases (2-4 words)
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        query_phrases = [' '.join(query_lower.split()[i:i+phrase_len]) 
                                        for i in range(len(query_lower.split()) - phrase_len + 1)]
                        for phrase in query_phrases:
                            if len(phrase.split()) >= 2 and phrase in page_text:
                                score += 3 * page_text.count(phrase)
            
            # Bonus for tables if query suggests data interest
            table_terms = {"table", "column", "field", "value", "schema", "mapping"}
            if any(term in query_lower for term in table_terms) and "TABLE:" in content["content"]:
                table_count = content["content"].count("TABLE:")
                score += table_count * 7  # Bonus for each table
            
            # Bonus for code examples if query suggests implementation interest
            code_terms = {"code", "example", "implementation", "syntax", "usage"}
            if any(term in query_lower for term in code_terms) and "```" in content["content"]:
                code_count = content["content"].count("```") // 2  # Each block has opening and closing
                score += code_count * 5  # Bonus for each code block
            
            # Check for relevant image descriptions
            image_terms = {"image", "diagram", "screenshot", "picture"}
            if any(term in query_lower for term in image_terms) and "[IMAGE:" in content["content"]:
                image_count = content["content"].count("[IMAGE:")
                score += image_count * 3  # Bonus for each image
            
            scored_pages.append((page, content, score))
        
        # Sort by score and take top results
        scored_pages.sort(key=lambda x: x[2], reverse=True)
        top_pages = scored_pages[:5]  # Take top 5 most relevant pages
        
        logger.info(f"Selected {len(top_pages)} most relevant pages")
        
        if not top_pages:
            return "I couldn't find any relevant information in the Confluence space."
        
        # Step 3: Extract relevant sections from top pages
        relevant_content = []
        
        for page, content, score in top_pages:
            page_content = content["content"]
            page_url = content["metadata"]["url"]
            
            # Split content into sections for more targeted extraction
            # Try to split by headings first
            sections = re.split(r'#{1,6}\s+', page_content)
            
            if len(sections) <= 2:  # If not many headings, use paragraphs
                sections = page_content.split("\n\n")
            
            # Score each section
            section_scores = []
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                section_lower = section.lower()
                section_score = 0
                
                # Score based on query terms
                for word in query_words:
                    freq = section_lower.count(word)
                    section_score += freq * 0.5
                
                # Extra points for exact phrase matches
                if query_lower in section_lower:
                    section_score += 10
                else:
                    # Check phrases
                    for phrase_len in range(2, 5):
                        if len(query_lower.split()) >= phrase_len:
                            phrases = [' '.join(query_lower.split()[i:i+phrase_len]) 
                                      for i in range(len(query_lower.split()) - phrase_len + 1)]
                            for phrase in phrases:
                                if phrase in section_lower:
                                    section_score += 2
                
                # Special handling for tables and code
                if "TABLE:" in section:
                    section_score *= 1.5  # Tables are usually highly relevant
                if "```" in section:
                    section_score *= 1.3  # Code examples are valuable
                
                section_scores.append((i, section, section_score))
            
            # Get top scoring sections (up to 3 from each page)
            section_scores.sort(key=lambda x: x[2], reverse=True)
            top_sections = section_scores[:3]
            
            # Order sections by their original position in the document
            ordered_sections = sorted(top_sections, key=lambda x: x[0])
            
            if ordered_sections:
                content_block = f"--- FROM: {page['title']} ---\n\n"
                
                # If first section doesn't start with a heading, add the page title as heading
                first_section = ordered_sections[0][1].strip()
                if not first_section.startswith('#'):
                    content_block += f"# {page['title']}\n\n"
                
                for _, section, _ in ordered_sections:
                    # Clean up the section
                    cleaned_section = re.sub(r'\n{3,}', '\n\n', section.strip())
                    content_block += cleaned_section + "\n\n"
                
                content_block += f"Source: {page_url}\n"
                relevant_content.append(content_block)
        
        # Combine relevant content from all pages
        return "\n\n" + "\n\n".join(relevant_content)
    
    def answer_question(self, question):
        """Answer a question using Confluence content and Gemini."""
        logger.info(f"Processing question: {question}")
        
        # Extract relevant content based on the question
        relevant_content = self.extract_relevant_content(question)
        
        # Generate response using Gemini
        response = self.gemini.generate_response(question, relevant_content)
        
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
    print(f"I've loaded information from {len(assistant.space_pages)} pages in the {CONFLUENCE_SPACE} space.")
    print("I can help you understand how COPPER database views map to REST APIs.")
    print("What would you like to know about COPPER views or APIs?")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Thanks for using the COPPER Assistant. Have a great day!")
                break
                
            if not user_input:
                continue
                
            print("\nLooking that up for you...")
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
