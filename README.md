#!/usr/bin/env python3
import logging
import os
import sys
import re
import json
import time
import html
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Vertex AI and Gemini imports
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

# Confluence API imports
import requests
from urllib.parse import quote
from html.parser import HTMLParser

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_confluence_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GeminiConfluenceChatbot")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CONFLUENCE_BASE_URL = os.environ.get("CONFLUENCE_BASE_URL", "https://your-company.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CACHE_EXPIRY_HOURS = int(os.environ.get("CACHE_EXPIRY_HOURS", "24"))  # Cache expiry in hours
MAX_CACHE_PAGES = int(os.environ.get("MAX_CACHE_PAGES", "10000"))  # Max pages to cache
PARALLEL_REQUESTS = int(os.environ.get("PARALLEL_REQUESTS", "5"))  # Number of parallel requests

class HTMLContentParser(HTMLParser):
    """HTML parser to extract clean text from HTML content including tables and images."""
    def __init__(self):
        super().__init__()
        self.text = ""
        self.current_tag = None
        self.in_table = False
        self.table_data = []
        self.current_row = []
        self.current_cell = ""
        self.table_counter = 0
        self.image_counter = 0
        self.images = []
        self.in_script = False
        self.in_style = False
        self.capture = True

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        if tag == 'script':
            self.in_script = True
            self.capture = False
        elif tag == 'style':
            self.in_style = True
            self.capture = False
        elif tag == 'table':
            self.in_table = True
            self.table_counter += 1
            self.table_data = []
        elif tag == 'tr' and self.in_table:
            self.current_row = []
        elif tag == 'td' or tag == 'th' and self.in_table:
            self.current_cell = ""
        elif tag == 'img':
            self.image_counter += 1
            if 'src' in attrs_dict and 'alt' in attrs_dict:
                self.images.append({
                    'src': attrs_dict['src'],
                    'alt': attrs_dict['alt'] if attrs_dict['alt'] else f"Image {self.image_counter}"
                })
                self.text += f"\n[Image: {attrs_dict['alt'] if attrs_dict['alt'] else f'Image {self.image_counter}'}]\n"
    
    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False
            self.capture = True
        elif tag == 'style':
            self.in_style = False
            self.capture = True
        elif tag == 'table':
            self.in_table = False
            # Add table representation to text
            self.text += f"\n[Table {self.table_counter}]\n"
            for row in self.table_data:
                self.text += "| " + " | ".join(row) + " |\n"
            self.text += "\n"
        elif tag == 'tr' and self.current_row:
            self.table_data.append(self.current_row)
        elif tag == 'td' or tag == 'th':
            self.current_row.append(self.current_cell.strip())
        elif tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']:
            self.text += "\n"
    
    def handle_data(self, data):
        if not self.capture:
            return
            
        if self.in_table and self.current_tag in ['td', 'th']:
            self.current_cell += data
        else:
            self.text += data
    
    def get_clean_text(self):
        # Clean up multiple newlines and other formatting issues
        text = re.sub(r'\n+', '\n', self.text).strip()
        # Decode HTML entities
        text = html.unescape(text)
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        return text

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "GeminiConfluenceChatbot/1.0 Python/Requests"
        }
        self.content_cache = {}  # Cache for page content
        self.space_cache = {}    # Cache for spaces
        self.cache_timestamp = None  # When the cache was last updated
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.base_url}/wiki/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (first connection): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received during connection test")
                return True  # Still consider it a success if status code is OK
            
            try:
                response.json()
                logger.info("Connection successful!")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Connection succeeded but received invalid JSON: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return False
        
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
            
            logger.info(f"Fetching content with ID: {content_id}, expand: {expand}")
            response = requests.get(
                f"{self.base_url}/wiki/rest/api/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (content by ID): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning(f"Empty response received when retrieving content ID: {content_id}")
                return None
            
            try:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
                return content
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for content ID {content_id}: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve content by ID: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                if hasattr(e, 'response') and e.response is not None and e.response.text:
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for processing.
        This extracts and processes the content to be more suitable for analysis.
        
        Args:
            page_id: The ID of the page
        """
        # Check the cache first
        if page_id in self.content_cache:
            logger.info(f"Using cached content for page ID: {page_id}")
            return self.content_cache[page_id]
            
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels,version")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])],
                "updated": page.get("version", {}).get("when"),
                "version": page.get("version", {}).get("number")
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process HTML content
            html_parser = HTMLContentParser()
            html_parser.feed(content)
            plain_text = html_parser.get_clean_text()
            
            processed_content = {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content,  # Include original HTML in case needed
                "images": html_parser.images
            }
            
            # Cache the processed content
            self.content_cache[page_id] = processed_content
            
            return processed_content
            
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def search_content(self, cql=None, title=None, content_type="page", expand=None, limit=10, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            expand: Properties to expand in results
            limit: Maximum number of results to return
            start: Starting index for pagination
        """
        try:
            params = {
                "limit": limit,
                "start": start
            }
            
            # Build CQL if not provided
            query_parts = []
            if content_type:
                query_parts.append(f"type={content_type}")
                
            if title:
                # Escape special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title~"{safe_title}"')
                
            if query_parts:
                params["cql"] = " AND ".join(query_parts)
            
            if cql:
                params["cql"] = cql
                
            if expand:
                params["expand"] = expand
                
            logger.info(f"Searching for content with params: {params}")
            response = requests.get(
                f"{self.base_url}/wiki/rest/api/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (search): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received when searching for content")
                return {"results": []}
            
            try:
                results = response.json()
                logger.info(f"Search returned {len(results.get('results', []))} results")
                return results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for search: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
                
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                if hasattr(e, 'response') and e.response is not None and e.response.text:
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
            return {"results": []}
    
    def get_all_content(self, content_type="page", limit=100):
        """Retrieve all content of specified type with pagination handling."""
        all_content = []
        start = 0
        
        logger.info(f"Retrieving all {content_type} content")
        
        while True:
            search_results = self.search_content(
                cql=f"type={content_type}",
                limit=limit,
                start=start
            )
            
            if not search_results:
                break
                
            results = search_results.get("results", [])
            if not results:
                break
                
            all_content.extend(results)
            logger.info(f"Retrieved {len(results)} {content_type} documents (total: {len(all_content)})")
            
            # Check if there are more pages
            if len(results) < limit:
                break
                
            # Increment for next page
            start += limit
            
            # Check the "_links" for a "next" link
            links = search_results.get("_links", {})
            if not links.get("next"):
                break
                
            # Avoid hitting rate limits
            time.sleep(0.5)
            
            # Safety check to avoid infinite loops
            if len(all_content) >= MAX_CACHE_PAGES:
                logger.warning(f"Reached maximum cache size of {MAX_CACHE_PAGES} pages")
                break
                
        logger.info(f"Retrieved a total of {len(all_content)} {content_type} documents")
        return all_content
    
    def get_spaces(self, limit=25, start=0):
        """
        Get all spaces the user has access to.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
        """
        try:
            params = {
                "limit": limit,
                "start": start
            }
            
            logger.info("Fetching spaces...")
            response = requests.get(
                f"{self.base_url}/wiki/rest/api/space",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (spaces): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received when fetching spaces")
                return {"results": []}
            
            try:
                spaces = response.json()
                logger.info(f"Successfully retrieved {len(spaces.get('results', []))} spaces")
                return spaces
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for spaces: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
                
        except requests.RequestException as e:
            logger.error(f"Failed to get spaces: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                if hasattr(e, 'response') and e.response is not None and e.response.text:
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
            return {"results": []}
    
    def get_all_spaces(self):
        """Retrieve all spaces with pagination handling."""
        if self.space_cache:
            logger.info(f"Using cached spaces ({len(self.space_cache)} spaces)")
            return list(self.space_cache.values())
            
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info("Retrieving all spaces")
        
        while True:
            spaces = self.get_spaces(limit=limit, start=start)
            if not spaces:
                break
                
            results = spaces.get("results", [])
            if not results:
                break
                
            all_spaces.extend(results)
            logger.info(f"Retrieved {len(results)} spaces")
            
            # Check if there are more pages
            if len(results) < limit:
                break
                
            # Increment for next page
            start += limit
            
            # Check the "_links" for a "next" link
            links = spaces.get("_links", {})
            if not links.get("next"):
                break
                
            # Avoid hitting rate limits
            time.sleep(0.5)
                
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        
        # Cache the spaces
        self.space_cache = {space.get("key"): space for space in all_spaces}
        return all_spaces
        
    def cache_all_content(self):
        """Cache all Confluence content for faster access."""
        # Check if cache is recent enough
        if self.cache_timestamp and (datetime.now() - self.cache_timestamp).total_seconds() < CACHE_EXPIRY_HOURS * 3600:
            logger.info(f"Using existing cache from {self.cache_timestamp}")
            return
            
        logger.info("Starting to cache all Confluence content")
        start_time = time.time()
        
        # Get all spaces first
        spaces = self.get_all_spaces()
        logger.info(f"Found {len(spaces)} spaces")
        
        # Get all content IDs first
        all_content = self.get_all_content(limit=100)
        logger.info(f"Found {len(all_content)} pages to cache")
        
        # Process content in parallel
        with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
            # Create a dictionary to map future to content_id for tracking
            future_to_id = {}
            
            # Submit processing tasks for each content item
            for content_item in all_content[:MAX_CACHE_PAGES]:
                content_id = content_item.get("id")
                if content_id:
                    future = executor.submit(self.get_page_content, content_id)
                    future_to_id[future] = content_id
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_id):
                content_id = future_to_id[future]
                try:
                    _ = future.result()
                    completed += 1
                    if completed % 20 == 0:
                        logger.info(f"Cached {completed}/{len(future_to_id)} pages")
                except Exception as e:
                    logger.error(f"Error processing content ID {content_id}: {str(e)}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Finished caching {len(self.content_cache)} pages in {elapsed_time:.2f} seconds")
        self.cache_timestamp = datetime.now()
    
    def search_cached_content(self, query, max_results=20):
        """
        Search through cached content using a text query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            list: A list of page content objects matching the query
        """
        if not self.content_cache:
            logger.warning("No cached content available for searching")
            return []
            
        logger.info(f"Searching cached content for: {query}")
        
        # Prepare search terms
        search_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        # Search through cached content
        matched_pages = []
        
        for page_id, page_content in self.content_cache.items():
            score = 0
            content = page_content.get("content", "").lower()
            metadata = page_content.get("metadata", {})
            title = metadata.get("title", "").lower()
            
            # Check for term matches
            for term in search_terms:
                # Title matches are highly relevant
                if term in title:
                    score += 10
                # Content matches
                if term in content:
                    term_count = content.count(term)
                    score += min(term_count, 20)
            
            # Only include pages with matches
            if score > 0:
                # Create a copy with score added
                result = page_content.copy()
                result["relevance_score"] = score
                matched_pages.append(result)
        
        # Sort by relevance score and limit results
        matched_pages.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        logger.info(f"Found {len(matched_pages)} matches in cached content")
        
        return matched_pages[:max_results]

class GeminiManager:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini model: {MODEL_NAME}")
    
    def analyze_question(self, question):
        """
        Analyze the question to determine if it's multi-part and extract subquestions.
        
        Args:
            question (str): The user's question
            
        Returns:
            list: A list of subquestions
        """
        logger.info(f"Analyzing question: {question}")
        
        # Generate a generation config with appropriate parameters
        generation_config = GenerationConfig(
            temperature=0.2,  # Low temperature for more deterministic outputs
            top_p=0.8,
            max_output_tokens=4096,
        )
        
        # Create the prompt for analyzing the question
        prompt = f"""
        # TASK: Question Analysis for Enterprise Knowledge Base Search
        
        ## OBJECTIVE
        Analyze the user's question with precision to identify if it contains multiple distinct inquiries that should be addressed separately when searching an enterprise knowledge base. This analysis is critical for providing comprehensive, accurate answers from technical documentation.
        
        ## INPUT ANALYSIS
        Carefully examine the following question from a technical professional in a corporate environment:
        
        USER QUESTION: "{question}"
        
        ## DETAILED INSTRUCTIONS
        1. First, identify the core information-seeking goals in the question
        2. Determine whether the question contains:
           - Multiple unrelated questions (e.g., "How do I configure X? Also, what's the policy on Y?")
           - A complex question with interdependent parts (e.g., "How does X work and how can I implement it with Y?")
           - A single focused question (e.g., "What's the procedure for X?")
           - Questions with conditional logic (e.g., "If X is configured, how should we handle Y?")
        3. For multi-part questions, separate them into distinct, searchable queries
        4. For complex but single questions, keep them together if the parts are strongly related
        5. Preserve technical terminology exactly as written, including case sensitivity
        6. Maintain product names, API references, and technical specifications precisely
        
        ## RESPONSE FORMAT
        Return ONLY a JSON array of strings, with each string representing a subquestion to be processed.
        Do not include explanations, preambles, or metadata.
        
        ## EXAMPLES
        Example 1: "How do I configure JWT authentication in our API gateway and what are the best practices for token handling?"
        Response: ["How do I configure JWT authentication in our API gateway?", "What are the best practices for token handling with JWT authentication?"]
        
        Example 2: "What's the procedure for deploying a new microservice?"
        Response: ["What's the procedure for deploying a new microservice?"]
        
        Example 3: "Can you explain the SSO implementation? I need to understand how it works with our existing identity provider and if we need to make changes to our configuration."
        Response: ["How does SSO implementation work with our existing identity provider?", "What changes are needed in our configuration for SSO implementation?"]
        """
        
        # Get the response from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        response_text = response.text
        logger.info(f"Question analysis response: {response_text}")
        
        # Try to parse the response as JSON
        try:
            subquestions = json.loads(response_text)
            if isinstance(subquestions, list) and all(isinstance(q, str) for q in subquestions):
                return subquestions
            else:
                logger.warning("Invalid response format from question analysis")
                return [question]
        except json.JSONDecodeError:
            logger.warning("Failed to parse question analysis response as JSON")
            # Fall back to returning the original question
            return [question]
    
    def _extract_search_terms(self, question):
        """
        Extract key search terms from the question.
        
        Args:
            question: The question to extract terms from
            
        Returns:
            list: A list of search terms
        """
        # Generate a generation config with appropriate parameters
        generation_config = GenerationConfig(
            temperature=0.2,  # Low temperature for more deterministic outputs
            top_p=0.8,
            max_output_tokens=1024,
        )
        
        # Create the prompt for extracting search terms
        prompt = f"""
        # TASK: Enterprise Knowledge Base Search Term Optimization
        
        ## OBJECTIVE
        Extract precise, high-value search terms from a technical question to optimize search results in an enterprise Confluence knowledge base. This is a mission-critical task where the quality of search terms directly impacts the accuracy and relevance of information retrieved.
        
        ## INPUT FOR ANALYSIS
        Technical question from enterprise user:
        "{question}"
        
        ## DETAILED EXTRACTION REQUIREMENTS
        1. PRIMARY EXTRACTION: Identify 3-7 high-value search terms that would yield the most relevant technical documentation
        
        2. TERM PRIORITIZATION (in order of importance):
           - Technical product/feature names (e.g., "Kubernetes", "SAML", "OAuth2")
           - Specific technical processes (e.g., "deployment pipeline", "authentication flow")
           - Industry-standard protocol/technology names (e.g., "JWT", "SSL", "microservices")
           - Enterprise-specific terminology (e.g., "SDLC", "compliance frameworks")
           - Technical action phrases (e.g., "configure load balancer", "implement authorization")
           - Error codes, API endpoints, function names (if present)
        
        3. OPTIMIZE FOR ENTERPRISE SEARCH:
           - Preserve exact technical terms (maintain exact capitalization: OAuth not oauth)
           - Include technical abbreviations and their expanded forms if relevant (e.g., "SSO", "Single Sign-On")
           - For multi-word concepts, include both the full phrase and key components
           - Include alternative technical terminology for the same concept when applicable
           - For error messages or specific codes, extract the most unique identifiable parts
        
        4. EXPLICITLY EXCLUDE:
           - Generic terms with low discrimination value ("system", "application", "process")
           - Common verbs and non-technical actions ("create", "setup", "manage")
           - Articles, prepositions, and common adjectives
           - Terms that would be company-standard across all documentation
        
        ## RESPONSE FORMAT
        Return ONLY a JSON array of strings, each representing an optimized search term.
        No preambles, explanations, or additional context.
        
        ## EXAMPLES
        Example 1 - Question: "How do I configure JWT authentication for our API gateway?"
        Output: ["JWT authentication", "API gateway", "configure JWT", "authentication configuration", "JWT", "API security"]
        
        Example 2 - Question: "What's the process for deploying a Spring Boot application to our Kubernetes cluster?"
        Output: ["Spring Boot deployment", "Kubernetes cluster", "Spring Boot", "Kubernetes", "deployment process", "container deployment", "K8s"]
        """
        
        # Get the response from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        response_text = response.text
        logger.info(f"Search terms extraction response: {response_text}")
        
        # Try to parse the response as JSON
        try:
            search_terms = json.loads(response_text)
            if isinstance(search_terms, list) and all(isinstance(term, str) for term in search_terms):
                return search_terms
            else:
                logger.warning("Invalid response format from search terms extraction")
                # Fall back to simple word extraction
                return self._simple_term_extraction(question)
        except json.JSONDecodeError:
            logger.warning("Failed to parse search terms response as JSON")
            # Fall back to simple word extraction
            return self._simple_term_extraction(question)
    
    def _simple_term_extraction(self, text):
        """Simple fallback method to extract terms from text."""
        # Remove common punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Extract unique words, filtering out common small words
        words = text.split()
        stopwords = {"the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "or", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must"}
        terms = [word for word in words if len(word) > 3 and word.lower() not in stopwords]
        # Return unique terms, up to 5
        unique_terms = list(set(terms))
        return unique_terms[:5]
    
    def search_relevant_content(self, confluence_client, question, max_results=100):
        """
        Search for content relevant to the question in Confluence.
        
        Args:
            confluence_client: The Confluence client
            question: The question to search for
            max_results: Maximum number of results to return
            
        Returns:
            list: A list of page content objects
        """
        logger.info(f"Searching for content relevant to: {question}")
        
        # Try cached search first
        cached_results = confluence_client.search_cached_content(question, max_results=max_results)
        if cached_results:
            logger.info(f"Found {len(cached_results)} relevant pages in cache")
            return cached_results
        
        # If no cached results or cache is empty, try API search
        # Generate search terms from the question
        search_terms = self._extract_search_terms(question)
        
        # Get original words from the question for additional matching
        original_words = set(re.findall(r'\b\w+\b', question.lower()))
        original_words = {w for w in original_words if len(w) > 3}
        
        # Build a CQL query
        cql_parts = []
        for term in search_terms:
            if term:
                # Escape any special characters in the term
                safe_term = term.replace('"', '\\"')
                cql_parts.append(f'text ~ "{safe_term}"')
        
        cql_query = " AND ".join(cql_parts) if cql_parts else None
        
        # Search for content
        search_results = confluence_client.search_content(
            cql=cql_query,
            content_type="page",
            limit=max_results
        )
        
        # If we got few results, try a broader search
        if len(search_results.get("results", [])) < 5 and search_terms:
            logger.info("Few results found with specific terms, trying broader search")
            broader_cql_parts = []
            for term in search_terms[:2]:  # Use just the top 2 terms
                if term:
                    safe_term = term.replace('"', '\\"')
                    broader_cql_parts.append(f'text ~ "{safe_term}"')
            
            if broader_cql_parts:
                broader_cql = " OR ".join(broader_cql_parts)
                broader_results = confluence_client.search_content(
                    cql=broader_cql,
                    content_type="page",
                    limit=max_results
                )
                
                # Combine results, avoiding duplicates
                result_ids = {r.get("id") for r in search_results.get("results", [])}
                for result in broader_results.get("results", []):
                    if result.get("id") not in result_ids:
                        search_results.get("results", []).append(result)
                        result_ids.add(result.get("id"))
        
        # Extract the content from each result
        page_contents = []
        for result in search_results.get("results", []):
            page_id = result.get("id")
            if page_id:
                page_content = confluence_client.get_page_content(page_id)
                if page_content:
                    # Add a relevance score based on term matching
                    page_content["relevance_score"] = self._calculate_relevance_score(page_content, search_terms, original_words)
                    page_contents.append(page_content)
        
        # Sort by relevance score
        page_contents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        logger.info(f"Retrieved {len(page_contents)} pages relevant to the question")
        return page_contents
    
    def _calculate_relevance_score(self, page_content, search_terms, original_words):
        """Calculate a relevance score for a page based on term matching."""
        score = 0
        content = page_content.get("content", "").lower()
        metadata = page_content.get("metadata", {})
        title = metadata.get("title", "").lower()
        
        # Score based on search terms
        for term in search_terms:
            term_lower = term.lower()
            # Title matches are highly relevant
            if term_lower in title:
                score += 10
            # Content matches
            term_count = content.count(term_lower)
            score += min(term_count, 20)  # Cap to avoid skewing for long documents
        
        # Score based on original question words
        for word in original_words:
            if word in title:
                score += 3
            word_count = content.count(word)
            score += min(word_count, 10) * 0.2
        
        # Bonus for recent or labeled content
        if "updated" in metadata and metadata.get("updated"):
            # Simple recency bonus
            score += 5
        
        if "labels" in metadata and metadata.get("labels"):
            # Bonus for pages with labels (typically more important)
            score += len(metadata.get("labels")) * 2
        
        return score
    
    def generate_answer(self, question, contexts, include_references=True):
        """
        Generate an answer to the question based on the provided contexts.
        
        Args:
            question: The user's question
            contexts: A list of context objects with content and metadata
            include_references: Whether to include references to source pages
            
        Returns:
            str: The generated answer
        """
        logger.info(f"Generating answer for: {question}")
        
        # Generate a generation config with appropriate parameters
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
        )
        
        # Prepare the context information
        context_text = ""
        references = []
        
        for i, ctx in enumerate(contexts, 1):
            content = ctx.get("content", "")
            metadata = ctx.get("metadata", {})
            title = metadata.get("title", f"Document {i}")
            url = metadata.get("url", "")
            
            # Add to the context text (truncate if needed)
            max_context_length = 24000  # Increased for more context
            if len(context_text) + len(content) > max_context_length:
                remaining_length = max_context_length - len(context_text)
                if remaining_length > 100:  # Only add if we can include a meaningful chunk
                    content = content[:remaining_length] + "..."
                else:
                    # We've reached capacity
                    logger.info(f"Reached context capacity after {i-1} documents")
                    break
            
            context_text += f"\n\n--- Document {i}: {title} ---\n{content}\n"
            
            # Add to references
            references.append({
                "title": title,
                "url": url
            })
        
        # If no contexts found, provide a special message
        if not contexts:
            context_text = "\n\n--- No relevant documentation found ---\n"
        
        # Create the prompt for generating the answer - modified to be more assertive
        prompt = f"""
        # ENTERPRISE KNOWLEDGE ASSISTANT: RESPONSE GENERATION PROTOCOL
        
        ## SYSTEM ROLE AND IDENTITY
        You are ATLAS (Advanced Technical Learning and Support), an enterprise-grade knowledge assistant with privileged access to confidential Confluence documentation. You embody these core attributes:
        - AUTHORITATIVE: You provide definitive guidance based on official documentation
        - PRECISE: You deliver exact, accurate information with technical precision
        - CONTEXTUAL: You understand the enterprise environment and technical infrastructure
        - PROFESSIONALLY WARM: You maintain a balance of technical authority with approachable language
        
        ## KEY DIRECTIVE: PRIORITIZE ANSWERS OVER CLARIFICATIONS
        Your PRIMARY objective is to provide useful answers based on available information. Even with incomplete context, you must:
        1. Make confident best-effort responses using whatever context is available
        2. Extrapolate reasonably from partial information
        3. Provide an answer that is useful even if not comprehensive
        4. AVOID asking for clarification unless absolutely no relevant information exists
        
        ## RESPONSE PARAMETERS
        As ATLAS, follow these strict response parameters:
        
        ### CONTENT SOURCES
        - Draw information EXCLUSIVELY from the provided Confluence documentation
        - Make reasonable inferences to connect partial information
        - When information is limited, clearly state what you DO know rather than focusing on gaps
        
        ### RESPONSE STRUCTURE
        - Begin with a direct, concise answer to the core question
        - Provide whatever information you CAN based on available documentation
        - For complex questions, structure information hierarchically
        - If appropriate, include a brief "Summary" section at the beginning for quick reference
        - Use appropriate technical depth based on the question's complexity
        - For procedural information, provide step-by-step sequences with clear ordering
        
        ### INFORMATION TYPES AND FORMATTING
        Transform the raw Confluence content into optimally formatted responses:
        
        #### For Procedural Information:
        - Present as numbered steps with clear, actionable instructions
        - Highlight important parameters, configuration options, or command flags
        - Include any prerequisites or system requirements at the beginning
        
        #### For Technical Concepts:
        - Begin with a clear definition
        - Follow with key characteristics, components, or distinctions
        - Use concise paragraphs with logical flow between concepts
        
        #### For Reference Information:
        - Organize using tables for multi-attribute information
        - For lists of items, use bullet points with consistent formatting
        - For properties or parameters, structure as definition lists
        
        #### For Code or Configuration Examples:
        - Maintain precise formatting with proper indentation
        - Include contextual comments to explain key aspects
        - Highlight critical lines or parameters
        
        ### LIMITED INFORMATION HANDLING
        When working with partial information:
        - Focus on what IS available in the documentation, providing that confidently
        - Use phrases like "Based on the available documentation..." rather than apologizing
        - Make reasonable technical inferences when connections between concepts are clear
        - Only at the end, briefly mention if additional information might be available elsewhere
        
        ### VISUAL AND STRUCTURAL ELEMENTS
        - Use markdown formatting extensively for readability
        - Create tables using markdown pipe notation for structured data
        - Use appropriate heading levels (##, ###) for logical sectioning
        - Use bold for emphasis on critical points
        - Use code blocks for commands, code, configuration files
        
        ### TONE AND LANGUAGE
        - Use confident, authoritative language for factual information
        - Maintain professional language while being conversational
        - Use first-person plural ("we") when discussing organizational practices
        - Project certainty and avoid hedging language whenever possible
        - If no relevant information exists at all, state that directly and suggest alternative approaches
        
        ## INFORMATION CONTEXT
        The following documentation excerpts have been extracted from the enterprise Confluence knowledge base:
        
        {context_text}
        
        ## USER INQUIRY
        Respond to the following user question using the information provided above. Provide a complete, useful answer even if the documentation doesn't cover every aspect of the question:
        
        "{question}"
        """
        
        # Get the response from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        answer = response.text
        
        # If references weren't included but should be, append them
        if include_references and "Documentation Sources" not in answer and "References" not in answer and contexts:
            answer += "\n\n## Documentation Sources\n\nThis response was compiled from the following official documentation sources:\n\n"
            for i, ref in enumerate(references, 1):
                answer += f"{i}. [{ref['title']}]({ref['url']})\n"
            
            answer += "\n*Note: Access to these resources may require appropriate authentication and authorization based on your role permissions.*"
        
        logger.info(f"Generated answer of length: {len(answer)}")
        return answer
    
    def is_followup_question(self, question, conversation_history):
        """
        Determine if the question is a follow-up to previous conversation.
        
        Args:
            question: The current question
            conversation_history: List of previous Q&A pairs
            
        Returns:
            bool: Whether this is a follow-up question
        """
        if not conversation_history:
            return False
            
        # Check for pronoun references and other linguistic markers
        pronoun_indicators = ["it", "this", "that", "they", "them", "these", "those", "their"]
        followup_phrases = ["what about", "how about", "tell me more", "and", "also", "what if", "why"]
        
        # Simple rule-based checks
        question_lower = question.lower()
        
        # Check for pronouns at the beginning
        for pronoun in pronoun_indicators:
            if question_lower.startswith(pronoun + " ") or f" {pronoun} " in question_lower:
                return True
                
        # Check for follow-up phrases
        for phrase in followup_phrases:
            if question_lower.startswith(phrase + " "):
                return True
                
        # More sophisticated check using Gemini
        if len(conversation_history) > 0:
            # Get the most recent conversation
            last_question, last_answer = conversation_history[-1]
            
            # Generate a generation config with appropriate parameters
            generation_config = GenerationConfig(
                temperature=0.2,  # Low temperature for more deterministic outputs
                top_p=0.8,
                max_output_tokens=1024,
            )
            
            # Create the prompt for determining if this is a follow-up
            prompt = f"""
            # CONVERSATION CONTINUITY ANALYSIS
            
            ## TASK DEFINITION
            Perform precise linguistic and contextual analysis to determine if the new question represents a continuation of the previous conversation thread that requires contextual information from the previous exchange for complete semantic understanding.
            
            ## ANALYSIS INPUTS
            
            ### CONVERSATION HISTORY
            Previous question: "{last_question}"
            
            Previous response: ```
            {last_answer}
            ```
            
            ### CURRENT INPUT
            New question: "{question}"
            
            ## ANALYTICAL FRAMEWORK
            Analyze using these specific continuity indicators:
            
            ### 1. LINGUISTIC CONTINUITY MARKERS
            - Presence of pronouns lacking clear antecedents (it, they, these, those, this)
            - Demonstrative references to previous content (that issue, those steps)
            - Elliptical constructions (omission of words assumed from context)
            - Comparative references (more, less, better, similar, different) without explicit objects
            - Discourse markers indicating continuation (also, additionally, furthermore, moreover)
            
            ### 2. SEMANTIC CONTINUITY INDICATORS
            - Referenced concepts that were introduced in previous exchange
            - Questions about implications, consequences, or extensions of previous information
            - Requests for elaboration on specific elements from previous response
            - Clarification requests about previous content
            - Questions building upon assumptions established in previous exchange
            
            ### 3. CONVERSATIONAL FLOW PATTERNS
            - Direct response to information presented in previous answer
            - Question that addresses a subcomponent of the previous topic
            - Exploration of related or dependent concept
            - Expression of confusion/need for clarification about previous answer
            - Request for alternative approaches to previously discussed solution
            
            ## DETERMINATION PROTOCOL
            1. Identify presence of ANY continuity markers from the frameworks above
            2. Assess whether the new question can be fully understood WITHOUT the previous exchange
            3. Determine if important context would be lost by treating question in isolation
            4. Make binary determination based on continuity requirements
            
            ## RESPONSE FORMAT
            Respond with EXACTLY one word: "Yes" or "No"
            - "Yes" = The question requires previous conversation context
            - "No" = The question can stand alone without previous context
            """
            
            # Get the response from Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            response_text = response.text.strip().lower()
            return "yes" in response_text
            
        return False

class GeminiConfluenceChatbot:
    """Main chatbot class integrating Confluence and Gemini."""
    
    def __init__(self, cache_on_startup=True):
        """Initialize the chatbot."""
        self.confluence_client = ConfluenceClient(
            base_url=CONFLUENCE_BASE_URL,
            username=CONFLUENCE_USERNAME,
            api_token=CONFLUENCE_API_TOKEN
        )
        self.gemini_manager = GeminiManager()
        self.conversation_history = []
        
        # Test connections
        if not self.confluence_client.test_connection():
            logger.error("Failed to connect to Confluence. Please check your credentials and connection.")
            raise ConnectionError("Could not connect to Confluence API")
            
        logger.info("GeminiConfluenceChatbot initialized successfully")
        
        # Cache all content if requested
        if cache_on_startup:
            self.confluence_client.cache_all_content()
    
    def process_question(self, question):
        """
        Process a user question and generate a response.
        
        Args:
            question: The user's question
            
        Returns:
            str: The chatbot's response
        """
        logger.info(f"Processing question: {question}")
        
        # Check if this is a follow-up question
        is_followup = self.gemini_manager.is_followup_question(question, self.conversation_history)
        
        if is_followup and self.conversation_history:
            # For follow-up questions, include previous context
            logger.info("Detected follow-up question, incorporating previous context")
            
            # Get the most recent conversation
            prev_question, prev_answer = self.conversation_history[-1]
            
            # Create a combined question for search purposes
            combined_question = f"{prev_question} {question}"
            
            # Analyze the combined question
            subquestions = self.gemini_manager.analyze_question(combined_question)
        else:
            # For new questions, analyze to check if it's multi-part
            subquestions = self.gemini_manager.analyze_question(question)
        
        logger.info(f"Question broken down into {len(subquestions)} subquestions")
        
        # Collect contexts for all subquestions
        all_contexts = []
        all_references = set()  # Use a set to avoid duplicates
        
        for subq in subquestions:
            logger.info(f"Processing subquestion: {subq}")
            
            # Search for relevant content with higher max_results
            contexts = self.gemini_manager.search_relevant_content(
                self.confluence_client, 
                subq, 
                max_results=100 // len(subquestions)  # Divide the quota among subquestions
            )
            
            # Add contexts to the collected list, avoiding duplicates
            for ctx in contexts:
                metadata = ctx.get("metadata", {})
                doc_id = metadata.get("id")
                
                if doc_id and doc_id not in all_references:
                    all_references.add(doc_id)
                    all_contexts.append(ctx)
        
        logger.info(f"Collected {len(all_contexts)} unique contexts across all subquestions")
        
        # Generate an answer even if we have few or no contexts
        # We'll let the model do its best with whatever information we have
        answer = self.gemini_manager.generate_answer(question, all_contexts)
        
        # Add to conversation history
        self.conversation_history.append((question, answer))
        
        # Limit history to recent conversations
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
            
        return answer

    def chat_api(self, question):
        """
        Process a question and return the answer (API version).
        
        Args:
            question: The user's question
            
        Returns:
            dict: Response containing the answer and metadata
        """
        try:
            answer = self.process_question(question)
            return {
                "status": "success",
                "answer": answer,
                "conversation_id": id(self)  # Simple conversation ID for tracking
            }
        except Exception as e:
            logger.error(f"Error in chat_api: {str(e)}")
            return {
                "status": "error",
                "message": f"An error occurred: {str(e)}",
                "conversation_id": id(self)
            }
    
    def chat(self):
        """Run an interactive chat session in the console."""
        print("Welcome to the Gemini Confluence Chatbot!")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("=" * 50)
        
        while True:
            question = input("\nYou: ").strip()
            
            if question.lower() in ["exit", "quit", "bye"]:
                print("\nChatbot: Goodbye! Have a great day.")
                break
                
            if not question:
                continue
                
            try:
                print("\nChatbot is thinking...")
                answer = self.process_question(question)
                print(f"\nChatbot: {answer}")
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("\nChatbot: I encountered an error while processing your question. Please try again or rephrase your question.")

def create_flask_app(chatbot):
    """
    Create a Flask app for the chatbot API.
    
    Args:
        chatbot: The GeminiConfluenceChatbot instance
        
    Returns:
        Flask app
    """
    from flask import Flask, request, jsonify, render_template_string
    
    app = Flask(__name__)
    
    @app.route('/api/chat', methods=['POST'])
    def chat_endpoint():
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"status": "error", "message": "Missing 'question' in request"}), 400
        
        response = chatbot.chat_api(data['question'])
        return jsonify(response)
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "message": "Chatbot API is running"})
    
    @app.route('/', methods=['GET'])
    def index():
        # Simple HTML interface for the chatbot
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gemini Confluence Chatbot</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                #chat-container {
                    height: 500px;
                    overflow-y: scroll;
                    border: 1px solid #ccc;
                    padding: 10px;
                    margin-bottom: 10px;
                }
                .user-message {
                    background-color: #e1f5fe;
                    padding: 8px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    max-width: 80%;
                    margin-left: auto;
                }
                .bot-message {
                    background-color: #f5f5f5;
                    padding: 8px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    max-width: 80%;
                }
                #input-container {
                    display: flex;
                }
                #user-input {
                    flex-grow: 1;
                    padding: 8px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                #send-button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    margin-left: 10px;
                    cursor: pointer;
                }
                #send-button:hover {
                    background-color: #45a049;
                }
                pre {
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <h1>Gemini Confluence Chatbot</h1>
            <div id="chat-container"></div>
            <div id="input-container">
                <input type="text" id="user-input" placeholder="Ask a question..." />
                <button id="send-button">Send</button>
            </div>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const chatContainer = document.getElementById('chat-container');
                    const userInput = document.getElementById('user-input');
                    const sendButton = document.getElementById('send-button');
                    
                    // Function to add a message to the chat
                    function addMessage(message, isUser) {
                        const messageElement = document.createElement('div');
                        messageElement.className = isUser ? 'user-message' : 'bot-message';
                        
                        if (isUser) {
                            messageElement.textContent = message;
                        } else {
                            // Convert markdown to HTML (simple version)
                            let html = message
                                .replace(/\n\n## (.*?)\n/g, '<h2>$1</h2>')
                                .replace(/\n\n### (.*?)\n/g, '<h3>$1</h3>')
                                .replace(/\n\n/g, '<br><br>')
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                .replace(/```([\s\S]*?)```/g, '<pre>$1</pre>')
                                .replace(/`(.*?)`/g, '<code>$1</code>');
                                
                            messageElement.innerHTML = html;
                        }
                        
                        chatContainer.appendChild(messageElement);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                    
                    // Function to send a message to the API
                    function sendMessage() {
                        const message = userInput.value.trim();
                        if (!message) return;
                        
                        // Add user message to chat
                        addMessage(message, true);
                        userInput.value = '';
                        
                        // Add a loading message
                        const loadingElement = document.createElement('div');
                        loadingElement.className = 'bot-message';
                        loadingElement.textContent = 'Thinking...';
                        chatContainer.appendChild(loadingElement);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                        
                        // Send request to API
                        fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ question: message })
                        })
                        .then(response => response.json())
                        .then(data => {
                            // Remove loading message
                            chatContainer.removeChild(loadingElement);
                            
                            // Add bot message to chat
                            if (data.status === 'success') {
                                addMessage(data.answer, false);
                            } else {
                                addMessage('Sorry, I encountered an error processing your request.', false);
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            // Remove loading message
                            chatContainer.removeChild(loadingElement);
                            // Add error message
                            addMessage('Sorry, there was an error communicating with the server.', false);
                        });
                    }
                    
                    // Event listeners
                    sendButton.addEventListener('click', sendMessage);
                    userInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            sendMessage();
                        }
                    });
                    
                    // Add welcome message
                    addMessage('Welcome! I\'m your Confluence knowledge assistant. How can I help you today?', false);
                });
            </script>
        </body>
        </html>
        """
        return render_template_string(html)
    
    return app

def show_setup_instructions():
    """Display setup instructions if environment variables are missing."""
    print("\n" + "=" * 80)
    print("Gemini Confluence Chatbot - Setup Required")
    print("=" * 80)
    print("\nThe following environment variables need to be set:")
    print("  CONFLUENCE_BASE_URL: The base URL of your Confluence instance")
    print("  CONFLUENCE_USERNAME: Your Confluence username/email")
    print("  CONFLUENCE_API_TOKEN: Your Confluence API token")
    print("\nOptional variables:")
    print("  PROJECT_ID: Your Google Cloud project ID")
    print("  REGION: Google Cloud region for Vertex AI")
    print("  MODEL_NAME: Gemini model to use")
    print("  CACHE_EXPIRY_HOURS: How long to cache content (default: 24)")
    print("  MAX_CACHE_PAGES: Maximum number of pages to cache (default: 10000)")
    print("  PARALLEL_REQUESTS: Number of parallel requests for caching (default: 5)")
    print("\nExample setup (Bash):")
    print('  export CONFLUENCE_BASE_URL="https://your-company.atlassian.net"')
    print('  export CONFLUENCE_USERNAME="your.email@company.com"')
    print('  export CONFLUENCE_API_TOKEN="your-api-token"')
    print("\nTo get a Confluence API token:")
    print("  1. Log in to your Atlassian account")
    print("  2. Go to Account Settings → Security → Create and manage API tokens")
    print("  3. Create a new API token and copy it")
    print("\nAfter setting these variables, run the script again.")
    print("=" * 80 + "\n")

# Main execution
if __name__ == "__main__":
    try:
        # Check for required environment variables
        if not CONFLUENCE_BASE_URL or not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN:
            show_setup_instructions()
            sys.exit(1)
            
        # Check for API mode flag
        api_mode = len(sys.argv) > 1 and sys.argv[1] == "--api"
        no_cache = len(sys.argv) > 1 and sys.argv[1] == "--no-cache"
        
        # Create the chatbot
        chatbot = GeminiConfluenceChatbot(cache_on_startup=not no_cache)
        
        if api_mode:
            # Start the Flask API
            app = create_flask_app(chatbot)
            port = int(os.environ.get("PORT", 5000))
            print(f"\nStarting chatbot API server on port {port}...")
            print(f"Web interface: http://localhost:{port}/")
            print("Press Ctrl+C to stop the server")
            app.run(host="0.0.0.0", port=port)
        else:
            # Start the interactive console chat
            chatbot.chat()
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        sys.exit(1)





























#!/usr/bin/env python3
"""
Enterprise Remedy AI Assistant - An integrated solution for querying BMC Remedy incidents
using Gemini AI that provides comprehensive information to third-party users.

This system allows users to query Remedy incident data using natural language,
with AI-enhanced responses that provide summaries, details, and insights.
"""

import json
import requests
import logging
import os
import sys
import urllib3
from datetime import datetime, timedelta
import time
import getpass
import re
import textwrap
from urllib.parse import quote

# Import Vertex AI related libraries
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Disable SSL warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("remedy_ai_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RemedyAI")

# Configuration (Environment Variables or Default Values)
# IMPORTANT: Update the default REMEDY_SERVER to match your company's endpoint
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-001")
REMEDY_SERVER = os.environ.get("REMEDY_SERVER", "https://remedy-api.yourcompany.com")
MAX_INCIDENTS = int(os.environ.get("MAX_INCIDENTS", "50"))  # Maximum incidents to retrieve in a query

class RemedyClient:
    """
    Enhanced client for BMC Remedy REST API operations with comprehensive error handling,
    optimized for enterprise environments.
    """
    def __init__(self, server_url=REMEDY_SERVER, username=None, password=None, ssl_verify=False):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server
            username: Username for authentication
            password: Password for authentication
            ssl_verify: Whether to verify SSL certificates (default False for enterprise environments)
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None 
        self.token_type = "AR-JWT"
        self.ssl_verify = ssl_verify
        self.last_login_time = None
        self.session = requests.Session()  # Use session for connection pooling
        
        logger.info(f"Initialized Remedy client for {self.server_url}")

    def login(self, force=False):
        """
        Log in to Remedy and get authentication token with improved error handling
        and session management.
        
        Args:
            force: Force new login even if token exists

        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        # Check if we already have a valid token and not forcing relogin
        if not force and self.token and self.last_login_time:
            # Tokens typically expire after 1 hour, check if still valid
            elapsed_minutes = (datetime.now() - self.last_login_time).total_seconds() / 60
            if elapsed_minutes < 55:  # Renew before expiry
                logger.debug("Using existing token (still valid)")
                return 1, self.token
        
        # Get credentials if not provided
        if not self.username:
            self.username = input("Enter Remedy Username: ")
        if not self.password:
            self.password = getpass.getpass(prompt="Enter Remedy Password: ")
        
        logger.info(f"Attempting to login as {self.username}")
        url = f"{self.server_url}/api/jwt/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"content-type": "application/x-www-form-urlencoded"}
        
        try:
            r = self.session.post(url, data=payload, headers=headers, verify=self.ssl_verify, timeout=30)
            if r.status_code == 200:
                self.token = r.text
                self.last_login_time = datetime.now()
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                err_msg = f"Login failure... Status Code: {r.status_code}"
                if r.text:
                    err_msg += f", Response: {r.text[:200]}"
                print(err_msg)
                return -1, r.text
        except requests.RequestException as e:
            logger.error(f"Login connection error: {str(e)}")
            return -1, f"Connection error: {str(e)}"
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return -1, str(e)

    def ensure_authenticated(self):
        """
        Ensure we have a valid authentication token, refreshing if necessary.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        status, _ = self.login()
        return status == 1

    def get_incident(self, incident_id):
        """
        Get a specific incident by its ID with enhanced error handling.
        
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
            
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self.ensure_authenticated():
            logger.error("Authentication failed. Cannot retrieve incident.")
            return None
            
        logger.info(f"Fetching incident: {incident_id}")
        
        # Normalize incident ID format (add INC prefix if missing)
        if not incident_id.upper().startswith("INC"):
            if incident_id.isdigit() and len(incident_id) >= 9:
                incident_id = f"INC{incident_id}"
        
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Comprehensive fields list for detailed information
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager", "Last Modified Date",
            "Resolved Date", "Resolution", "Detailed Description", "Notes",
            "Closure Product Category Tier 1", "Closure Product Name", "Closure Product Model/Version",
            "Resolution Category", "Resolution Category Tier 2", "Service Type",
            "Reported Source", "Manufacturer", "Product Categorization Tier 1",
            "Product Categorization Tier 2", "Product Categorization Tier 3", "Product Name"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            
            # Also get the incident history for more context
            history = self.get_incident_history(incident_id)
            
            return {
                "incident": result["entries"][0],
                "history": history
            }
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None

    def get_incident_history(self, incident_id):
        """
        Get the history of changes for a specific incident.
        
        Args:
            incident_id: The Incident Number
            
        Returns:
            list: History entries or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        logger.info(f"Fetching history for incident: {incident_id}")
        
        # Build URL for history form
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk History"
        
        # Qualified query to filter by incident number
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Query parameters
        params = {
            "q": qualified_query,
            "fields": "History Date Time,Action,Description,Status,Changed By,Assigned Group,Details,Previous Value,New Value",
            "sort": "History Date Time DESC"  # Sort by date descending
        }
        
        # Make the request
        try:
            r = self.session.get(url, headers=headers, params=params, verify=self.ssl_verify, timeout=30)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully retrieved history for incident {incident_id} with {len(result.get('entries', []))} entries")
                return result.get("entries", [])
            else:
                logger.error(f"Get history failed with status code: {r.status_code}")
                return []
        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
            return []

    def get_incidents_by_date_range(self, start_date, end_date=None, status=None, owner_group=None, limit=MAX_INCIDENTS):
        """
        Get all incidents submitted within a date range.
        
        Args:
            start_date: Start date (datetime object)
            end_date: End date (datetime object, defaults to start_date + 1 day)
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        if not end_date:
            # If no end date provided, use 1 day after start date
            end_date = start_date + timedelta(days=1)
            
        logger.info(f"Fetching incidents from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Format datetime strings for query
        start_datetime = start_date.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = end_date.strftime("%Y-%m-%d 00:00:00.000")
        
        # Create qualified query
        query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
        
        # Add status filter if provided
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        # Add owner group filter if provided
        if owner_group:
            query_parts.append(f"'Owner Group'=\"{owner_group}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents for date range")
            return result["entries"]
        else:
            logger.warning(f"No incidents found for date range or error occurred")
            return []

    def get_incidents_by_date(self, date_str, status=None, owner_group=None, limit=MAX_INCIDENTS):
        """
        Get all incidents submitted on a specific date with enhanced date parsing.
        
        Args:
            date_str: Date string (can be 'today', 'yesterday', 'YYYY-MM-DD', etc.)
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        # Parse the date string to get a datetime object
        date_obj = self._parse_date_expression(date_str)
        if not date_obj:
            logger.error(f"Invalid date format or expression: {date_str}")
            return []
            
        # Use the date range function with single day range
        return self.get_incidents_by_date_range(date_obj, None, status, owner_group, limit)

    def get_incidents_by_status(self, status, limit=MAX_INCIDENTS):
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        logger.info(f"Fetching incidents with status: {status}")
        
        # Normalize status (capitalize first letter of each word)
        status = ' '.join(word.capitalize() for word in status.split())
        
        # Create qualified query
        qualified_query = f"'Status'=\"{status}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []

    def get_incidents_by_text(self, search_text, fields=None, limit=MAX_INCIDENTS):
        """
        Search for incidents containing specific text in selected fields.
        
        Args:
            search_text: Text to search for
            fields: List of field names to search in (default: Summary & Description)
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        if not fields:
            fields = ["Summary", "Description"]
            
        logger.info(f"Searching for incidents with text '{search_text}' in fields {fields}")
        
        # Create qualified query for text search with LIKE operator
        query_parts = []
        for field in fields:
            query_parts.append(f"'{field}' LIKE \"%{search_text}%\"")
            
        # Join conditions with OR for broader search
        qualified_query = " OR ".join(query_parts)
        
        # Fields to retrieve
        output_fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, output_fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents matching text search")
            return result["entries"]
        else:
            logger.warning(f"No incidents found matching text search or error occurred")
            return []

    def get_incidents_by_assignee(self, assignee, status=None, limit=MAX_INCIDENTS):
        """
        Get incidents assigned to a specific person.
        
        Args:
            assignee: The assignee name
            status: Optional status filter
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        logger.info(f"Fetching incidents assigned to: {assignee}")
        
        # Create qualified query
        query_parts = [f"'Assignee' LIKE \"%{assignee}%\""]  # Use LIKE for partial matches
        if status:
            status = ' '.join(word.capitalize() for word in status.split())
            query_parts.append(f"'Status'=\"{status}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents assigned to {assignee}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found assigned to {assignee} or error occurred")
            return []

    def get_incidents_by_owner_group(self, owner_group, status=None, limit=MAX_INCIDENTS):
        """
        Get incidents owned by a specific group.
        
        Args:
            owner_group: The owner group name
            status: Optional status filter
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        logger.info(f"Fetching incidents owned by group: {owner_group}")
        
        # Create qualified query with LIKE for partial matches
        query_parts = [f"'Owner Group' LIKE \"%{owner_group}%\""]
        if status:
            status = ' '.join(word.capitalize() for word in status.split())
            query_parts.append(f"'Status'=\"{status}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents owned by group {owner_group}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found owned by group {owner_group} or error occurred")
            return []

    def get_incidents_by_priority(self, priority, status=None, limit=MAX_INCIDENTS):
        """
        Get incidents by priority level.
        
        Args:
            priority: Priority level (e.g., "Critical", "High", "Medium", "Low")
            status: Optional status filter
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        logger.info(f"Fetching incidents with priority: {priority}")
        
        # Create qualified query
        query_parts = [f"'Priority' LIKE \"%{priority}%\""]
        if status:
            status = ' '.join(word.capitalize() for word in status.split())
            query_parts.append(f"'Status'=\"{status}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with priority {priority}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with priority {priority} or error occurred")
            return []

    def get_recent_incidents(self, days=7, status=None, limit=MAX_INCIDENTS):
        """
        Get recent incidents from the past X days.
        
        Args:
            days: Number of days back to look
            status: Optional status filter
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        logger.info(f"Fetching recent incidents from past {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format datetime strings for query
        start_datetime = start_date.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = end_date.strftime("%Y-%m-%d 23:59:59.999")
        
        # Create qualified query
        query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' <= \"{end_datetime}\""]
        
        # Add status filter if provided
        if status:
            status = ' '.join(word.capitalize() for word in status.split())
            query_parts.append(f"'Status'=\"{status}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} recent incidents")
            return result["entries"]
        else:
            logger.warning("No recent incidents found or error occurred")
            return []

    def query_form(self, form_name, qualified_query=None, fields=None, limit=100, sort=None):
        """
        Query a Remedy form with optional filters and field selection.
        
        Args:
            form_name: The name of the form to query (e.g., "HPD:Help Desk")
            qualified_query: Optional qualified query string for filtering
            fields: Optional list of fields to retrieve
            limit: Maximum number of records to retrieve
            sort: Optional sort field and direction (e.g., "Submit Date DESC")
            
        Returns:
            dict: Query result or None if error
        """
        if not self.ensure_authenticated():
            return None
            
        logger.info(f"Querying form: {form_name}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/{form_name}"
        
        # Build headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Build query parameters
        params = {}
        if qualified_query:
            params["q"] = qualified_query
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
        if sort:
            params["sort"] = sort
        
        # Make the request with retry logic
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                r = self.session.get(url, headers=headers, params=params, verify=self.ssl_verify, timeout=30)
                if r.status_code == 200:
                    result = r.json()
                    logger.info(f"Successfully queried form {form_name} and got {len(result.get('entries', []))} results")
                    return result
                elif r.status_code == 401 and retry_count < max_retries:
                    # Token might have expired, try to re-login
                    logger.warning("Authentication token expired, attempting to refresh")
                    self.login(force=True)
                    headers = {"Authorization": f"{self.token_type} {self.token}"}
                else:
                    logger.error(f"Query failed with status code: {r.status_code}")
                    logger.error(f"Response: {r.text[:200]}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Query connection error: {str(e)}")
                if retry_count < max_retries:
                    logger.info(f"Retrying query (attempt {retry_count+1}/{max_retries})")
                else:
                    return None
            except Exception as e:
                logger.error(f"Query error: {str(e)}")
                return None
                
            retry_count += 1
            time.sleep(1)  # Short delay before retry
            
        return None

    def _parse_date_expression(self, date_str):
        """
        Parse a date expression into a datetime object.
        Supports formats:
        - 'today', 'yesterday'
        - 'N days ago'
        - 'last week', 'last month'
        - 'YYYY-MM-DD'
        - 'MM/DD/YYYY'
        
        Args:
            date_str: Date expression as string
            
        Returns:
            datetime: Parsed date object or None if invalid
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check for relative date expressions
        date_str = date_str.lower().strip()
        
        if date_str == 'today':
            return today
        elif date_str == 'yesterday':
            return today - timedelta(days=1)
        elif date_str == 'last week':
            # Start of the previous week (Monday)
            days_since_monday = today.weekday()
            return today - timedelta(days=days_since_monday + 7)
        elif date_str == 'this week':
            # Start of the current week (Monday)
            days_since_monday = today.weekday()
            return today - timedelta(days=days_since_monday)
        elif date_str == 'last month':
            # Approximate a month as 30 days
            return today - timedelta(days=30)
        
        # Check for "N days ago" pattern
        days_ago_match = re.match(r'(\d+)\s+days?\s+ago', date_str)
        if days_ago_match:
            days = int(days_ago_match.group(1))
            return today - timedelta(days=days)
        
        # Try to parse YYYY-MM-DD
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
        
        # Try to parse MM/DD/YYYY
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            pass
        
        # If we get here, we couldn't parse the date
        return None


class GeminiHelper:
    """
    Helper class for interacting with Google's Gemini AI model,
    optimized for analyzing Remedy incident data.
    """
    
    def __init__(self, project_id=PROJECT_ID, location=REGION, model_name=MODEL_NAME):
        """
        Initialize the Gemini helper.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Gemini model name
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.model = None
            
    def generate_response(self, prompt, temperature=0.2):
        """
        Generate a response from Gemini.
        
        Args:
            prompt: The prompt text
            temperature: Generation temperature (0-1)
            
        Returns:
            str: The generated response
        """
        if not self.model:
            return "Error: Gemini not properly initialized."
            
        logger.info(f"Generating Gemini response with {len(prompt)} characters")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            if hasattr(response, 'text'):
                logger.info(f"Successfully generated response with {len(response.text)} characters")
                return response.text
            else:
                logger.warning("Empty or invalid response from Gemini")
                return "I couldn't generate a response for that query. Please try again with a more specific question."
                
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return f"I encountered an error when analyzing this data: {str(e)}"

    def analyze_incidents(self, query, incidents, single_incident=False, history=None):
        """
        Analyze incidents data and provide AI-enhanced response.
        
        Args:
            query: User's original query
            incidents: List of incidents (or single incident) to analyze
            single_incident: Whether this is a single incident query
            history: Optional incident history data
            
        Returns:
            str: The AI-generated analysis
        """
        # Build a comprehensive prompt
        prompt_parts = [
            "You are an expert IT service management assistant specializing in BMC Remedy.",
            "You provide clear, concise, and comprehensive responses about incident tickets.",
            "",
            f"USER QUERY: {query}",
            ""
        ]
        
        # Add incident data based on type of query
        if single_incident and incidents:
            # Single incident with detailed analysis
            prompt_parts.append("INCIDENT DETAILS:")
            incident_values = incidents["values"]
            
            # Add all available fields
            for field, value in incident_values.items():
                if value:  # Only include fields with values
                    prompt_parts.append(f"{field}: {value}")
            
            # Add history if available
            if history and len(history) > 0:
                prompt_parts.append("\nINCIDENT HISTORY:")
                for entry in history:
                    if "values" in entry:
                        history_values = entry["values"]
                        history_entry = [
                            f"Time: {history_values.get('History Date Time', 'N/A')}",
                            f"Action: {history_values.get('Action', 'N/A')}",
                            f"By: {history_values.get('Changed By', 'N/A')}",
                            f"Description: {history_values.get('Description', 'N/A')}"
                        ]
                        prompt_parts.append(" | ".join(history_entry))
        
        elif incidents and len(incidents) > 0:
            # Multiple incidents summary
            prompt_parts.append(f"INCIDENTS DATA ({len(incidents)} incidents):")
            
            # Add a summary of each incident
            for i, incident in enumerate(incidents):
                if "values" not in incident:
                    continue
                    
                values = incident["values"]
                prompt_parts.append(f"\n--- Incident {i+1} ---")
                incident_fields = [
                    f"Incident Number: {values.get('Incident Number', 'N/A')}",
                    f"Summary: {values.get('Summary', 'N/A')}",
                    f"Status: {values.get('Status', 'N/A')}",
                    f"Priority: {values.get('Priority', 'N/A')}",
                    f"Assignee: {values.get('Assignee', 'N/A')}",
                    f"Owner Group: {values.get('Owner Group', 'N/A')}",
                    f"Submit Date: {values.get('Submit Date', 'N/A')}"
                ]
                
                if "Description" in values:
                    incident_fields.append(f"Description: {values['Description']}")
                    
                prompt_parts.append("\n".join(incident_fields))
        else:
            prompt_parts.append("NO INCIDENTS FOUND MATCHING THE QUERY.")
        
        # Add instructions for response generation
        prompt_parts.append("\nINSTRUCTIONS:")
        prompt_parts.append("1. Provide a direct answer to the user's query based on the incident data.")
        prompt_parts.append("2. Format your response appropriately (use tables, bullet points, etc. as needed).")
        prompt_parts.append("3. Include specific incident numbers when referencing tickets.")
        prompt_parts.append("4. Do NOT ask follow-up questions - provide the best possible answer with available information.")
        prompt_parts.append("5. Be concise but thorough, focusing on what the user asked about.")
        prompt_parts.append("6. If analyzing multiple incidents, provide helpful patterns or insights.")
        prompt_parts.append("7. If no incidents were found, suggest alternative search approaches.")
        
        # Add special instruction for different query types
        if single_incident:
            prompt_parts.append("8. For this single incident query, provide a comprehensive analysis including status, key events, and next steps if applicable.")
        elif incidents and len(incidents) > 5:
            prompt_parts.append(f"8. You're analyzing {len(incidents)} incidents, so focus on providing a statistical summary rather than details of each incident.")
        
        # Complete the prompt
        prompt_parts.append("\nYour response to the user (provide ONLY the content the user should see):")
        
        # Join all prompt parts
        full_prompt = "\n".join(prompt_parts)
        
        # Generate the response
        return self.generate_response(full_prompt)


class RemedyAIAssistant:
    """
    Main class for the Enterprise Remedy AI Assistant that integrates
    Remedy and Gemini to provide comprehensive incident information to users.
    """
    
    def __init__(self, remedy_username=None, remedy_password=None):
        """
        Initialize the AI assistant.
        
        Args:
            remedy_username: Optional username for Remedy authentication
            remedy_password: Optional password for Remedy authentication
        """
        self.remedy = RemedyClient(username=remedy_username, password=remedy_password, ssl_verify=False)
        self.gemini = GeminiHelper()
        
        # Track conversation context
        self.last_incidents = []
        self.last_query = None
        
        logger.info("Enterprise Remedy AI Assistant initialized")
    
    def login(self):
        """Login to Remedy."""
        status, _ = self.remedy.login()
        return status == 1
    
    def detect_query_type(self, query_text):
        """
        Detect the type of query and extract relevant parameters.
        
        Args:
            query_text: The user's query text
            
        Returns:
            tuple: (query_type, params) with extracted parameters
        """
        query_text = query_text.strip().lower()
        
        # Pattern for incident number - high priority match
        inc_match = re.search(r'\b(?:inc\s*)?(\d{9,})\b', query_text, re.IGNORECASE)
        inc_match2 = re.search(r'\b(inc\d{9,})\b', query_text, re.IGNORECASE)
        
        if inc_match2:
            return "incident", {"incident_id": inc_match2.group(1)}
        elif inc_match:
            return "incident", {"incident_id": inc_match.group(1)}
        
        # Check for date-based queries
        date_patterns = [
            (r'\b(today|yesterday|last\s+week|this\s+week)\b', "date"),
            (r'\b(\d{1,2}/\d{1,2}/\d{4})\b', "date"),
            (r'\b(\d{4}-\d{1,2}-\d{1,2})\b', "date"),
            (r'\b(\d+)\s+days?\s+ago\b', "date")
        ]
        
        for pattern, query_type in date_patterns:
            date_match = re.search(pattern, query_text)
            if date_match:
                return query_type, {"date": date_match.group(1)}
        
        # Check for status-based queries
        status_match = re.search(r'\b(open|closed|pending|resolved|in\s*progress)\s+incidents\b', query_text)
        if status_match:
            return "status", {"status": status_match.group(1)}
        
        # Check for assignee queries
        assignee_patterns = [
            r'(?:assigned|assign)(?:ed)?\s+to\s+([a-zA-Z\s]+)',
            r'tickets?\s+for\s+([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)(?:\'s)?\s+tickets'
        ]
        
        for pattern in assignee_patterns:
            assignee_match = re.search(pattern, query_text)
            if assignee_match:
                assignee = assignee_match.group(1).strip()
                if len(assignee) > 3:  # Avoid matching short words
                    return "assignee", {"assignee": assignee}
        
        # Check for owner group queries
        group_patterns = [
            r'(?:owned|supported)\s+by\s+(?:group|team)\s+([a-zA-Z\s]+)',
            r'tickets?\s+for\s+(?:group|team)\s+([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)\s+(?:group|team)(?:\'s)?\s+tickets'
        ]
        
        for pattern in group_patterns:
            group_match = re.search(pattern, query_text)
            if group_match:
                group = group_match.group(1).strip()
                if len(group) > 3:  # Avoid matching short words
                    return "group", {"group": group}
        
        # Check for priority-based queries
        priority_match = re.search(r'\b(critical|high|medium|low)\s+priority\b', query_text)
        if priority_match:
            return "priority", {"priority": priority_match.group(1)}
        
        # Check for text search queries
        search_patterns = [
            r'(?:search|find)\s+(?:for|incidents|tickets)?\s+(?:with|containing|about)\s+(?:text|keyword)?\s*[\'"]?([^\'"]+)[\'"]?',
            r'incidents\s+(?:with|containing|about)\s+(?:text|keyword)?\s*[\'"]?([^\'"]+)[\'"]?'
        ]
        
        for pattern in search_patterns:
            search_match = re.search(pattern, query_text)
            if search_match:
                search_text = search_match.group(1).strip()
                if len(search_text) > 2:  # Avoid matching very short search terms
                    return "search", {"text": search_text}
        
        # Check for recent/summary queries
        if re.search(r'\b(recent|latest|new)\s+incidents\b', query_text):
            return "recent", {"days": 7}
        
        if re.search(r'\b(all|summary|total)\s+incidents\b', query_text):
            return "summary", {"limit": 20}
        
        # If no specific pattern matched, treat as general text search
        # Extract significant words from the query by removing common stop words
        stop_words = ['the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 
                     'to', 'from', 'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 
                     'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                     'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 
                     'should', 'may', 'might', 'must', 'that', 'this', 'these', 'those', 'i', 
                     'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
        
        words = query_text.split()
        significant_words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
        
        if significant_words:
            # Join the first 2-3 significant words as a search term
            search_text = " ".join(significant_words[:min(len(significant_words), 3)])
            return "search", {"text": search_text}
        
        # Default to general query
        return "general", {}
    
    def process_query(self, query_text):
        """
        Process a user query and generate a comprehensive response.
        
        Args:
            query_text: The user's question or command
            
        Returns:
            str: The AI-enhanced response
        """
        logger.info(f"Processing query: {query_text}")
        
        # Ensure we're authenticated
        if not self.remedy.ensure_authenticated():
            return "I'm having trouble connecting to the Remedy system. Please check the connection and credentials."
        
        try:
            # Detect the type of query
            query_type, params = self.detect_query_type(query_text)
            logger.info(f"Detected query type: {query_type} with params: {params}")
            
            # Fetch data based on query type
            incidents = []
            history = None
            
            if query_type == "incident":
                # Get a specific incident
                incident_data = self.remedy.get_incident(params["incident_id"])
                if incident_data:
                    incidents = incident_data["incident"]
                    history = incident_data["history"]
                    # Generate response using Gemini
                    response = self.gemini.analyze_incidents(query_text, incidents, single_incident=True, history=history)
                else:
                    response = f"I couldn't find incident {params['incident_id']} in the system. Please verify the incident number and try again."
            
            elif query_type == "date":
                # Get incidents for a specific date
                incidents = self.remedy.get_incidents_by_date(params["date"])
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "status":
                # Get incidents by status
                incidents = self.remedy.get_incidents_by_status(params["status"])
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "assignee":
                # Get incidents by assignee
                incidents = self.remedy.get_incidents_by_assignee(params["assignee"])
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "group":
                # Get incidents by owner group
                incidents = self.remedy.get_incidents_by_owner_group(params["group"])
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "priority":
                # Get incidents by priority
                incidents = self.remedy.get_incidents_by_priority(params["priority"])
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "search":
                # Search for incidents containing specific text
                incidents = self.remedy.get_incidents_by_text(params["text"])
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "recent":
                # Get recent incidents
                days = params.get("days", 7)
                incidents = self.remedy.get_recent_incidents(days)
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            elif query_type == "summary":
                # Get a summary of incidents
                limit = params.get("limit", 20)
                incidents = self.remedy.get_recent_incidents(30, limit=limit)  # Last 30 days
                self.last_incidents = incidents  # Store for context
                response = self.gemini.analyze_incidents(query_text, incidents)
            
            else:
                # General query - use Gemini to analyze with context
                if self.last_incidents:
                    # Use the last fetched incidents as context
                    response = self.gemini.analyze_incidents(query_text, self.last_incidents)
                else:
                    # No context, get recent incidents as default context
                    incidents = self.remedy.get_recent_incidents(7, limit=10)  # Last week, up to 10
                    self.last_incidents = incidents  # Store for context
                    response = self.gemini.analyze_incidents(query_text, incidents)
            
            # Store the last query for context
            self.last_query = query_text
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}"
    
    def run_chat_loop(self):
        """
        Run an interactive chat loop for the AI assistant.
        """
        print("\n" + "=" * 50)
        print("Enterprise Remedy AI Assistant")
        print("Ask questions about incidents in your Remedy system.")
        print("Type 'exit', 'quit', or 'bye' to end the session.")
        print("=" * 50 + "\n")
        
        # Login to Remedy
        if not self.login():
            print("Failed to authenticate with Remedy. Please check your credentials.")
            return
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Enterprise Remedy AI Assistant. Goodbye!")
                break
            
            print("\nProcessing your query...")
            response = self.process_query(user_input)
            
            print("\nAssistant:")
            # Print response with proper formatting
            for line in response.split('\n'):
                print(textwrap.fill(line, width=100) if len(line) > 100 else line)


if __name__ == "__main__":
    # Get credentials from environment or user input
    remedy_username = os.environ.get("REMEDY_USERNAME")
    remedy_password = os.environ.get("REMEDY_PASSWORD")
    
    # Create and run the AI assistant
    assistant = RemedyAIAssistant(remedy_username, remedy_password)
    assistant.run_chat_loop()




























#!/usr/bin/env python3
"""
Simplified Remedy Gemini Chatbot - A direct integration between BMC Remedy and Gemini AI.
"""

import json
import requests
import logging
import os
import sys
import urllib3
from datetime import datetime, timedelta
import time
import getpass
import re
from urllib.parse import quote
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Disable SSL warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("remedy_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RemedyChatbot")

# Configuration (Environment Variables or Default Values)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-001")
REMEDY_SERVER = os.environ.get("REMEDY_SERVER", "https://cmegroup-restapi.onbmc.com")

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling and
    advanced querying.
    """
    def __init__(self, server_url=REMEDY_SERVER, username=None, password=None, ssl_verify=False):
        """
        Initialize the Remedy client with server and authentication details.
        Args:
            server_url: The base URL of the Remedy server (e.g., https://cmegroup-restapi.onbmc.com)
            username: Username for authentication (will prompt if None)
            password: Password for authentication (will prompt if None)
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None 
        self.token_type = "AR-JWT"
        self.ssl_verify = ssl_verify
        
        logger.info(f"Initialized Remedy client for {self.server_url}")

    def login(self):
        """
        Log in to Remedy and get authentication token.
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        if not self.username:
            self.username = input("Enter Username: ")
        if not self.password:
            self.password = getpass.getpass(prompt="Enter Password: ")
        
        logger.info(f"Attempting to login as {self.username}")
        url = f"{self.server_url}/api/jwt/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"content-type": "application/x-www-form-urlencoded"}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=self.ssl_verify)
            if r.status_code == 200:
                self.token = r.text
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                print(f"Failure...")
                print(f"Status Code: {r.status_code}")
                return -1, r.text
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return -1, str(e)

    def get_incident(self, incident_id):
        """
        Get a specific incident by its ID.
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
            
        logger.info(f"Fetching incident: {incident_id}")
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve - comprehensive list
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager", "Last Modified Date",
            "Resolved Date", "Resolution", "Detailed Description"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None

    def get_incidents_by_date(self, date_str, status=None, owner_group=None):
        """
        Get all incidents submitted on a specific date with enhanced date parsing.
        Args:
            date_str: Date string (can be 'today', 'yesterday', 'YYYY-MM-DD', etc.)
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
            
        # Parse the date string
        date_obj = self._parse_date_string(date_str)
        if not date_obj:
            logger.error(f"Could not parse date: {date_str}")
            return []
            
        logger.info(f"Fetching incidents for date: {date_obj.strftime('%Y-%m-%d')}")
        
        # Create date range (entire day)
        start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
        
        # Create qualified query
        query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
        
        # Add status filter if provided
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        # Add owner group filter if provided
        if owner_group:
            query_parts.append(f"'Owner Group'=\"{owner_group}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents for date {date_obj.strftime('%Y-%m-%d')}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found for date {date_obj.strftime('%Y-%m-%d')} or error occurred")
            return []

    def get_incidents_by_status(self, status, limit=50):
        """
        Get incidents by their status.
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents with status: {status}")
        
        # Create qualified query
        qualified_query = f"'Status'=\"{status}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []
            
    def get_incidents_by_text(self, search_text, limit=50):
        """
        Search for incidents containing specific text in Summary or Description.
        Args:
            search_text: Text to search for
            limit: Maximum number of incidents to retrieve
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Searching for incidents with text: {search_text}")
        
        # Create qualified query for text search in Summary or Description
        qualified_query = f"'Summary' LIKE \"%{search_text}%\" OR 'Description' LIKE \"%{search_text}%\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents matching text search")
            return result["entries"]
        else:
            logger.warning(f"No incidents found matching text search or error occurred")
            return []

    def get_incidents_by_assignee(self, assignee, limit=50):
        """
        Get incidents assigned to a specific person.
        Args:
            assignee: The assignee name
            limit: Maximum number of incidents to retrieve
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents assigned to: {assignee}")
        
        # Create qualified query
        qualified_query = f"'Assignee'=\"{assignee}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents assigned to {assignee}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found assigned to {assignee} or error occurred")
            return []

    def get_incidents_by_owner_group(self, owner_group, limit=50):
        """
        Get incidents owned by a specific group.
        Args:
            owner_group: The owner group name
            limit: Maximum number of incidents to retrieve
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents owned by group: {owner_group}")
        
        # Create qualified query
        qualified_query = f"'Owner Group'=\"{owner_group}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents owned by group {owner_group}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found owned by group {owner_group} or error occurred")
            return []

    def query_form(self, form_name, qualified_query=None, fields=None, limit=100):
        """
        Query a Remedy form with optional filters and field selection.
        Args:
            form_name: The name of the form to query (e.g., "HPD:Help Desk")
            qualified_query: Optional qualified query string for filtering
            fields: Optional list of fields to retrieve
            limit: Maximum number of records to retrieve
        Returns:
            dict: Query result or None if error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
            
        logger.info(f"Querying form: {form_name}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/{form_name}"
        
        # Build headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Build query parameters
        params = {}
        if qualified_query:
            params["q"] = qualified_query
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully queried form {form_name} and got {len(result.get('entries', []))} results")
                return result
            else:
                logger.error(f"Query failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return None

    def _parse_date_string(self, date_str):
        """
        Parse a date string into a datetime object.
        Supports: 'today', 'yesterday', 'YYYY-MM-DD', 'MM/DD/YYYY'
        
        Args:
            date_str: The date string to parse
            
        Returns:
            datetime: Parsed date or None if parsing failed
        """
        date_str = date_str.lower().strip()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Handle special date strings
        if date_str == 'today':
            return today
        elif date_str == 'yesterday':
            return today - timedelta(days=1)
        elif date_str == 'last week':
            return today - timedelta(days=7)
        
        # Handle "N days ago" pattern
        match = re.match(r'(\d+)\s+days?\s+ago', date_str)
        if match:
            days = int(match.group(1))
            return today - timedelta(days=days)
        
        # Try parsing standard date formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                pass
        
        # If we get here, we couldn't parse the date
        return None


class GeminiHelper:
    """Helper class for Gemini API interactions."""
    
    def __init__(self, project_id=PROJECT_ID, location=REGION, model_name=MODEL_NAME):
        """Initialize the Gemini helper."""
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.model = None
            
    def generate_response(self, prompt, temperature=0.2):
        """Generate a response from Gemini."""
        if not self.model:
            return "Error: Gemini not properly initialized."
            
        logger.info(f"Generating Gemini response for prompt: {prompt[:100]}...")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=8192
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if hasattr(response, 'text'):
                return response.text
            else:
                return "Error: No response text received from Gemini."
                
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return f"Error generating response: {str(e)}"


class RemedyChatbot:
    """Main chatbot class integrating Remedy and Gemini."""
    
    def __init__(self, remedy_username=None, remedy_password=None):
        """Initialize the chatbot."""
        self.remedy = RemedyClient(username=remedy_username, password=remedy_password, ssl_verify=False)
        self.gemini = GeminiHelper()
        self.chat_history = []
        
    def login(self):
        """Login to Remedy."""
        status, _ = self.remedy.login()
        return status == 1
        
    def format_incident(self, incident):
        """Format an incident for display."""
        if not incident or "values" not in incident:
            return "Incident data not available."
            
        values = incident["values"]
        
        # Format the basic incident information
        lines = [
            f"Incident Number: {values.get('Incident Number', 'N/A')}",
            f"Summary: {values.get('Summary', 'N/A')}",
            f"Status: {values.get('Status', 'N/A')}",
            f"Priority: {values.get('Priority', 'N/A')}",
            f"Assigned To: {values.get('Assignee', 'N/A')}",
            f"Owner Group: {values.get('Owner Group', 'N/A')}",
            f"Submitter: {values.get('Submitter', 'N/A')}",
            f"Submit Date: {values.get('Submit Date', 'N/A')}",
            f"Impact: {values.get('Impact', 'N/A')}",
            "",
            f"Description: {values.get('Description', 'N/A')}",
            ""
        ]
        
        # Add resolution if available
        if values.get('Resolution'):
            lines.append(f"Resolution: {values.get('Resolution')}")
            
        return "\n".join(lines)
        
    def format_incidents_summary(self, incidents):
        """Format a summary of multiple incidents."""
        if not incidents:
            return "No incidents found."
            
        # Format basic information for each incident
        lines = [f"Found {len(incidents)} incidents:"]
        
        for i, incident in enumerate(incidents, 1):
            if "values" not in incident:
                continue
                
            values = incident["values"]
            lines.append(f"{i}. {values.get('Incident Number', 'N/A')} - {values.get('Summary', 'N/A')}")
            lines.append(f"   Status: {values.get('Status', 'N/A')} | Priority: {values.get('Priority', 'N/A')} | Assignee: {values.get('Assignee', 'N/A')}")
            lines.append(f"   Submit Date: {values.get('Submit Date', 'N/A')}")
            lines.append("")
            
        return "\n".join(lines)
    
    def process_direct_query(self, query):
        """
        Process a specific query type directly without using Gemini for intent detection.
        This handles common query patterns directly.
        """
        # Try to match specific incident number pattern (INCXXXXXXXXX)
        inc_match = re.search(r'INC\d{9,}', query, re.IGNORECASE)
        if inc_match:
            incident_id = inc_match.group(0)
            logger.info(f"Direct query for incident: {incident_id}")
            
            incident = self.remedy.get_incident(incident_id)
            if incident:
                return f"Here are the details for {incident_id}:\n\n{self.format_incident(incident)}"
            else:
                return f"I couldn't find incident {incident_id} in the system."
        
        # Check for status queries
        status_match = re.search(r'(open|closed|in progress|pending|resolved)\s+incidents', query, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).title()  # Capitalize first letter
            logger.info(f"Direct query for {status} incidents")
            
            incidents = self.remedy.get_incidents_by_status(status, limit=10)
            return f"Here's a summary of {status} incidents:\n\n{self.format_incidents_summary(incidents)}"
        
        # Check for date queries
        date_patterns = [
            r'incidents\s+(?:from|on)\s+(yesterday|today|last week)',
            r'incidents\s+(?:from|on)\s+(\d{4}-\d{1,2}-\d{1,2})',
            r'incidents\s+(?:from|on)\s+(\d{1,2}/\d{1,2}/\d{4})',
            r'what\s+happened\s+(yesterday|today|last week)'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, query, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                logger.info(f"Direct query for incidents on date: {date_str}")
                
                incidents = self.remedy.get_incidents_by_date(date_str)
                return f"Here's a summary of incidents from {date_str}:\n\n{self.format_incidents_summary(incidents)}"
        
        # Check for assignee queries
        assignee_match = re.search(r'(?:incidents|tickets)\s+(?:assigned to|for)\s+([A-Za-z\s]+)', query, re.IGNORECASE)
        if assignee_match:
            assignee = assignee_match.group(1).strip()
            logger.info(f"Direct query for incidents assigned to: {assignee}")
            
            incidents = self.remedy.get_incidents_by_assignee(assignee, limit=10)
            return f"Here's a summary of incidents assigned to {assignee}:\n\n{self.format_incidents_summary(incidents)}"
        
        # Check for owner group queries
        group_match = re.search(r'(?:incidents|tickets)\s+(?:owned by|for) (?:group|team)\s+([A-Za-z\s]+)', query, re.IGNORECASE)
        if group_match:
            group = group_match.group(1).strip()
            logger.info(f"Direct query for incidents owned by group: {group}")
            
            incidents = self.remedy.get_incidents_by_owner_group(group, limit=10)
            return f"Here's a summary of incidents owned by group {group}:\n\n{self.format_incidents_summary(incidents)}"
        
        # Text search queries
        search_patterns = [
            r'(?:search|find|look for)\s+incidents\s+(?:with|containing)\s+(?:text|keyword)?\s*["\']?([^"\']+)["\']?',
            r'incidents\s+(?:with|containing|about)\s+(?:text|keyword)?\s*["\']?([^"\']+)["\']?'
        ]
        
        for pattern in search_patterns:
            search_match = re.search(pattern, query, re.IGNORECASE)
            if search_match:
                search_text = search_match.group(1).strip()
                logger.info(f"Direct query for text search: {search_text}")
                
                incidents = self.remedy.get_incidents_by_text(search_text, limit=10)
                return f"Here's a summary of incidents containing '{search_text}':\n\n{self.format_incidents_summary(incidents)}"
        
        # If no direct pattern matched, try using Gemini to find what might be useful
        return None
    
    def process_with_gemini(self, query, incidents=None):
        """
        Process a query using Gemini, optionally with incident data.
        """
        # Build the prompt for Gemini
        prompt_parts = [
            "You are an expert IT service management assistant helping with BMC Remedy incidents.",
            "Please analyze the following information and provide a helpful, concise response.",
            "",
            f"USER QUERY: {query}",
            ""
        ]
        
        # Add incidents data if provided
        if incidents:
            prompt_parts.append(f"INCIDENTS DATA ({len(incidents)} incidents):")
            for i, incident in enumerate(incidents, 1):
                if "values" not in incident:
                    continue
                
                values = incident["values"]
                prompt_parts.append(f"--- Incident {i} ---")
                for key, value in values.items():
                    prompt_parts.append(f"{key}: {value}")
                prompt_parts.append("")
        
        # Add instructions for Gemini
        prompt_parts.append("\nINSTRUCTIONS:")
        prompt_parts.append("1. Provide a clear, concise response to the user's question.")
        prompt_parts.append("2. Format your answer appropriately (e.g., use bullet points for lists).")
        prompt_parts.append("3. Reference specific incident numbers when relevant.")
        prompt_parts.append("4. If there's not enough data to answer the question fully, say so clearly.")
        prompt_parts.append("5. Don't ask follow-up questions - just provide the best answer with the information available.")
        prompt_parts.append("\nYour response (provide ONLY the content the user should see):")
        
        # Get response from Gemini
        full_prompt = "\n".join(prompt_parts)
        return self.gemini.generate_response(full_prompt)
    
    def process_query(self, query):
        """Process a user query and generate a response."""
        logger.info(f"Processing query: {query}")
        
        # First, try to handle the query directly
        direct_result = self.process_direct_query(query)
        if direct_result:
            return direct_result
        
        # If no direct processing, use a simple keyword-based approach
        # to fetch relevant data for Gemini
        incidents = []
        
        # Check for common keywords and fetch relevant data
        keywords = query.lower().split()
        
        # If query mentions specific statuses
        if any(status in keywords for status in ['open', 'closed', 'resolved', 'pending']):
            for status in ['Open', 'Closed', 'Resolved', 'Pending']:
                if status.lower() in keywords:
                    incidents.extend(self.remedy.get_incidents_by_status(status, limit=5))
        
        # If query mentions dates
        date_keywords = ['today', 'yesterday', 'week']
        if any(date in keywords for date in date_keywords):
            for date in date_keywords:
                if date in keywords:
                    if date == 'today':
                        incidents.extend(self.remedy.get_incidents_by_date('today'))
                    elif date == 'yesterday':
                        incidents.extend(self.remedy.get_incidents_by_date('yesterday'))
                    elif date == 'week':
                        incidents.extend(self.remedy.get_incidents_by_date('last week'))
        
        # If query mentions looking for text
        if any(term in keywords for term in ['search', 'find', 'containing', 'about']):
            # Try to extract significant words (nouns) from the query
            # Exclude common stop words and just use remaining significant words
            stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'for', 'to', 'in', 'with', 'by', 'on', 'at']
            search_terms = [word for word in keywords if word not in stop_words 
                            and len(word) > 3 
                            and word not in ['search', 'find', 'incident', 'incidents', 'ticket', 'tickets']]
            
            if search_terms:
                for term in search_terms[:2]:  # Use up to 2 search terms
                    incidents.extend(self.remedy.get_incidents_by_text(term, limit=3))
        
        # Deduplicate incidents based on incident number
        unique_incidents = []
        incident_numbers = set()
        for incident in incidents:
            if "values" in incident and "Incident Number" in incident["values"]:
                incident_number = incident["values"]["Incident Number"]
                if incident_number not in incident_numbers:
                    incident_numbers.add(incident_number)
                    unique_incidents.append(incident)
        
        # If we have relevant incidents, use them for Gemini's response
        # If not, still use Gemini but without specific incident data
        return self.process_with_gemini(query, unique_incidents if unique_incidents else None)
    
    def run_chat_loop(self):
        """Run an interactive chat loop."""
        print("\n" + "=" * 50)
        print("Remedy Chatbot")
        print("Ask questions about your Remedy incidents.")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 50 + "\n")
        
        # Login to Remedy
        if not self.login():
            print("Failed to authenticate with Remedy. Please check your credentials.")
            return
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Remedy Chatbot. Goodbye!")
                break
            
            print("\nProcessing...")
            response = self.process_query(user_input)
            print("\nChatbot:", response.strip())


if __name__ == "__main__":
    # Get credentials from environment or user input
    remedy_username = os.environ.get("REMEDY_USERNAME")
    remedy_password = os.environ.get("REMEDY_PASSWORD")
    
    chatbot = RemedyChatbot(remedy_username, remedy_password)
    chatbot.run_chat_loop()
