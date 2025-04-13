#!/usr/bin/env python3
import logging
import os
import sys
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Vertex AI and Gemini imports
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

# Confluence API imports
import requests
from urllib.parse import quote
from html.parser import HTMLParser

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
MAX_RESULTS_PER_QUERY = 50  # Maximum number of Confluence pages to fetch per subquery

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

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        if tag == 'table':
            self.in_table = True
            self.table_counter += 1
            self.table_data = []
        elif tag == 'tr' and self.in_table:
            self.current_row = []
        elif tag == 'td' and self.in_table:
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
        if tag == 'table':
            self.in_table = False
            # Add table representation to text
            self.text += f"\n[Table {self.table_counter}]\n"
            for row in self.table_data:
                self.text += "| " + " | ".join(row) + " |\n"
            self.text += "\n"
        elif tag == 'tr' and self.current_row:
            self.table_data.append(self.current_row)
        elif tag == 'td':
            self.current_row.append(self.current_cell.strip())
        elif tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']:
            self.text += "\n"
    
    def handle_data(self, data):
        if self.in_table and self.current_tag == 'td':
            self.current_cell += data
        else:
            self.text += data
    
    def get_clean_text(self):
        return re.sub(r'\n+', '\n', self.text).strip()

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
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.base_url}/rest/api/content",
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
                f"{self.base_url}/rest/api/content/{content_id}",
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
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/spaces/{page.get('_links', {}).get('space', '')}/pages/{page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process HTML content
            html_parser = HTMLContentParser()
            html_parser.feed(content)
            plain_text = html_parser.get_clean_text()
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content  # Include original HTML in case needed
            }
            
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
                f"{self.base_url}/rest/api/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=True
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
    
    def get_all_content(self, content_type="page", limit=50):
        """Retrieve all content of specified type with pagination handling."""
        all_content = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
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
            logger.info(f"Retrieved {len(results)} {content_type} documents")
            
            # Check if there are more pages
            if len(results) < limit:
                break
                
            # Increment for next page
            start += limit
            
            # Check the "_links" for a "next" link
            links = search_results.get("_links", {})
            if not links.get("next"):
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
                f"{self.base_url}/rest/api/space",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=True
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
                
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces

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
    
    def search_relevant_content(self, confluence_client, question, max_results=MAX_RESULTS_PER_QUERY):
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
        
        # Generate search terms from the question
        search_terms = self._extract_search_terms(question)
        
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
        
        # Extract the content from each result
        page_contents = []
        for result in search_results.get("results", []):
            page_id = result.get("id")
            if page_id:
                page_content = confluence_client.get_page_content(page_id)
                if page_content:
                    page_contents.append(page_content)
        
        logger.info(f"Retrieved {len(page_contents)} pages relevant to the question")
        return page_contents
    
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
            max_context_length = 8000  # Adjust based on model's context window
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
        
        # Create the prompt for generating the answer
        prompt = f"""
        # ENTERPRISE KNOWLEDGE ASSISTANT: RESPONSE GENERATION PROTOCOL
        
        ## SYSTEM ROLE AND IDENTITY
        You are ATLAS (Advanced Technical Learning and Support), an enterprise-grade knowledge assistant with privileged access to confidential Confluence documentation. You embody these core attributes:
        - AUTHORITATIVE: You provide definitive guidance based exclusively on official documentation
        - PRECISE: You deliver exact, accurate information with technical precision
        - CONTEXTUAL: You understand the enterprise environment and technical infrastructure
        - PROFESSIONALLY WARM: You maintain a balance of technical authority with approachable language
        
        ## RESPONSE PARAMETERS
        As ATLAS, follow these strict response parameters:
        
        ### CONTENT SOURCES
        - Draw information EXCLUSIVELY from the provided Confluence documentation
        - NEVER fabricate or assume information not present in the documentation
        - Prioritize newer documentation over older when indicated by dates
        - When multiple sources provide conflicting information, state the conflict transparently
        
        ### RESPONSE STRUCTURE
        - Begin with a direct, concise answer to the core question
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
        
        ### EDGE CASE HANDLING
        
        #### For incomplete information:
        - Provide what is available in the documentation
        - Explicitly state what information appears to be missing
        - Suggest logical next steps or alternative approaches when appropriate
        
        #### For ambiguous questions:
        - Provide the most likely interpretation based on context
        - Acknowledge alternative interpretations
        - Ask a focused clarifying question
        
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
        - Avoid overly casual language, slang, or excessive technical jargon without explanation
        - For technical terms, provide brief explanations where appropriate
        
        ## INFORMATION CONTEXT
        The following documentation excerpts have been extracted from the enterprise Confluence knowledge base:
        
        {context_text}
        
        ## USER INQUIRY
        Respond to the following user question using only the information provided above:
        
        "{question}"
        """
        
        if include_references:
            prompt += """
            End your response with a "References" section that lists the titles and URLs of the Confluence pages you used to generate your answer.
            Format each reference as a numbered list item with the page title as a link.
            """
        
        # Get the response from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        answer = response.text
        
        # If references weren't included but should be, append them
        if include_references and "References" not in answer:
            answer += "\n\n## References\n"
            for i, ref in enumerate(references, 1):
                answer += f"{i}. [{ref['title']}]({ref['url']})\n"
        
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
    
    def generate_clarifying_questions(self, question, contexts):
        """
        Generate clarifying questions when the context doesn't provide a clear answer.
        
        Args:
            question: The user's question
            contexts: The retrieved contexts
            
        Returns:
            list: A list of clarifying questions
        """
        logger.info(f"Generating clarifying questions for: {question}")
        
        # Generate a generation config with appropriate parameters
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=2048,
        )
        
        # Prepare the context information (abbreviated version)
        context_text = ""
        for i, ctx in enumerate(contexts[:3], 1):  # Limit to first 3 contexts
            content = ctx.get("content", "")
            metadata = ctx.get("metadata", {})
            title = metadata.get("title", f"Document {i}")
            
            # Add a shorter version to the context text
            content_preview = content[:500] + "..." if len(content) > 500 else content
            context_text += f"\n\n--- Document {i}: {title} ---\n{content_preview}\n"
        
        # Create the prompt for generating clarifying questions
        prompt = f"""
        # ENTERPRISE KNOWLEDGE ASSISTANT: CLARIFICATION PROTOCOL
        
        ## SYSTEM CONTEXT
        You are ATLAS (Advanced Technical Learning and Support), an enterprise knowledge assistant specializing in technical documentation. You've been presented with a query where the available documentation is insufficient for a complete, accurate response. Your mission is to generate precision-targeted clarifying questions that will yield the exact information needed.
        
        ## CLARIFICATION OBJECTIVE
        Generate 1-3 strategic clarifying questions that will:
        1. Narrow the scope of ambiguity to specific technical parameters
        2. Eliminate potential misinterpretations of technical requirements
        3. Identify the exact information gaps between the question and available documentation
        4. Determine the specific technical context needed (environment, version, configuration)
        5. Establish the precise goal/outcome the user seeks (troubleshooting, implementation, optimization)
        
        ## QUESTION ENGINEERING SPECIFICATIONS
        
        ### TECHNICAL PRECISION
        - Focus on technical parameters, specifications, and configurations
        - Use precise technical terminology consistent with enterprise environment
        - Frame questions to distinguish between similar technologies, versions, or approaches
        
        ### INFORMATION ARCHITECTURE
        For each clarifying question:
        - Start with the most critical information gap
        - Progress to configuration/implementation specifics
        - End with goal/outcome clarification if needed
        
        ### CONTEXT OPTIMIZATION
        When formulating questions, prioritize clarifying:
        1. SYSTEM CONTEXT: Environment, deployment models, architecture
        2. TECHNICAL SPECIFICITY: Versions, technologies, protocols
        3. IMPLEMENTATION DETAILS: Configuration, parameters, constraints
        4. OUTCOME EXPECTATIONS: Success criteria, use cases, requirements
        
        ### PROFESSIONAL TONE CALIBRATION
        - Use concise, direct phrasing appropriate for technical professionals
        - Maintain confident, consultative tone
        - Avoid unnecessary apologies or hedging language
        - Project expertise while remaining approachable
        
        ## PRESENT CONTEXT
        User question: "{question}"
        
        Available documentation context (potentially incomplete):
        {context_text}
        
        ## RESPONSE FORMAT
        Return ONLY a JSON array of 1-3 strategically engineered clarifying questions.
        Do not include explanations, preambles, or apologies.
        
        ## EXAMPLES
        Example 1 - Context: Insufficient API authentication documentation
        Output: ["Which authentication protocol (OAuth2, JWT, or API keys) are you currently implementing?", "Are you integrating with our internal identity provider or an external service?", "Is this for production deployment or development/testing environment?"]
        
        Example 2 - Context: Incomplete deployment procedure
        Output: ["Which version of the application stack are you attempting to deploy?", "Are you deploying to our Kubernetes cluster or the legacy VM environment?"]
        """
        
        # Get the response from Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        response_text = response.text
        logger.info(f"Clarifying questions response: {response_text}")
        
        # Try to parse the response as JSON
        try:
            questions = json.loads(response_text)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
            else:
                logger.warning("Invalid response format from clarifying questions generation")
                return ["Could you provide more details about your question?"]
        except json.JSONDecodeError:
            logger.warning("Failed to parse clarifying questions response as JSON")
            # Try to extract questions using regex as a fallback
            questions = re.findall(r'(?:^|\n)(?:\d+[\.\)]\s*)?([^.\n]+\?)', response_text)
            if questions:
                return questions[:3]  # Limit to 3 questions
            else:
                return ["Could you provide more details about your question?"]

class GeminiConfluenceChatbot:
    """Main chatbot class integrating Confluence and Gemini."""
    
    def __init__(self):
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
            
            # Search for relevant content
            contexts = self.gemini_manager.search_relevant_content(
                self.confluence_client, 
                subq, 
                max_results=MAX_RESULTS_PER_QUERY // len(subquestions)  # Divide the quota among subquestions
            )
            
            # Add contexts to the collected list, avoiding duplicates
            for ctx in contexts:
                metadata = ctx.get("metadata", {})
                doc_id = metadata.get("id")
                
                if doc_id and doc_id not in all_references:
                    all_references.add(doc_id)
                    all_contexts.append(ctx)
        
        logger.info(f"Collected {len(all_contexts)} unique contexts across all subquestions")
        
        # If we have contexts, generate an answer
        if all_contexts:
            answer = self.gemini_manager.generate_answer(question, all_contexts)
            
            # Add to conversation history
            self.conversation_history.append((question, answer))
            
            # Limit history to recent conversations
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
                
            return answer
        else:
            # No contexts found, generate clarifying questions
            clarifying_questions = self.gemini_manager.generate_clarifying_questions(question, [])
            
            response = """## Information Gap Analysis

I've searched our enterprise knowledge base for information relevant to your inquiry, but I need additional details to provide you with a precise answer.

### Clarification Needed

To better address your specific needs, could you please provide more context on:

"""
            for i, q in enumerate(clarifying_questions, 1):
                response += f"{i}. {q}\n\n"
                
            response += """
### Next Steps

Your additional details will help me locate the most relevant documentation and provide you with accurate, actionable information. Alternatively, you might consider rephrasing your question with more specific technical details or context about your implementation environment."""
            
            # Save this interaction too
            self.conversation_history.append((question, response))
            
            # Limit history to recent conversations
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
                
            return response

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
    from flask import Flask, request, jsonify
    
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
    
    return app

# Main execution
if __name__ == "__main__":
    try:
        # Check for required environment variables
        if not CONFLUENCE_BASE_URL or not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN:
            logger.error("Missing required environment variables for Confluence integration")
            print("Please set the following environment variables:")
            print("- CONFLUENCE_BASE_URL: The base URL of your Confluence instance")
            print("- CONFLUENCE_USERNAME: Your Confluence username")
            print("- CONFLUENCE_API_TOKEN: Your Confluence API token")
            sys.exit(1)
            
        # Check for API mode flag
        api_mode = len(sys.argv) > 1 and sys.argv[1] == "--api"
        
        # Create the chatbot
        chatbot = GeminiConfluenceChatbot()
        
        if api_mode:
            # Start the Flask API
            app = create_flask_app(chatbot)
            port = int(os.environ.get("PORT", 5000))
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
Remedy Gemini Chatbot - An integrated solution for querying BMC Remedy incidents using Gemini AI.
This chatbot allows natural language queries about incidents, with context-aware responses and
the ability to understand tables, images, and provide references to relevant incidents.
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
from typing import List, Dict, Any, Optional, Tuple, Union
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Content
from google.api_core.exceptions import GoogleAPICallError

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
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-001")  # Using a more capable model
REMEDY_SERVER = os.environ.get("REMEDY_SERVER", "https://cmegroup-restapi.onbmc.com")

class RemedyClient:
    """
    Enhanced client for BMC Remedy REST API operations with comprehensive error handling,
    advanced querying, and persistent session management.
    """
    def __init__(self, server_url=REMEDY_SERVER, username=None, password=None, ssl_verify=False):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server
            username: Username for authentication (will prompt if None)
            password: Password for authentication (will prompt if None)
            ssl_verify: Whether to verify SSL certificates (default False)
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.token_type = "AR-JWT"
        self.ssl_verify = ssl_verify
        self.last_login_time = None
        self.token_expiry_minutes = 60  # Assuming tokens expire after 60 minutes
        
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self, force=False):
        """
        Log in to Remedy and get authentication token. Handles token expiry management.
        
        Args:
            force: Force a new login even if token exists
            
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        # Check if we already have a valid token
        if not force and self.token and self.last_login_time:
            elapsed_minutes = (datetime.now() - self.last_login_time).total_seconds() / 60
            if elapsed_minutes < self.token_expiry_minutes:
                logger.debug("Using existing token (still valid)")
                return 1, self.token
        
        # Get credentials if not provided
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
                self.last_login_time = datetime.now()
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                print(f"Login failure... Status Code: {r.status_code}")
                return -1, r.text
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
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve - comprehensive list
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager", "Last Modified Date",
            "Resolved Date", "Closure Manufacturer", "Closure Product Name", "Closure Product Model/Version",
            "Resolution", "Resolution Category", "Resolution Category Tier 2",
            "Closure Product Category Tier 1", "Detailed Description", "Notes"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
    
    def get_incidents_by_filter(self, query_parts, limit=100):
        """
        Get incidents based on a set of filter conditions.
        
        Args:
            query_parts: List of query conditions to be joined with AND
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_authenticated():
            return []
            
        qualified_query = " AND ".join(query_parts)
        logger.info(f"Querying incidents with filter: {qualified_query}")
        
        # Comprehensive fields list
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager", "Last Modified Date",
            "Resolved Date", "Closure Manufacturer", "Closure Product Name", "Resolution"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with the given filter")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with the given filter or error occurred")
            return []
    
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
        if not self.ensure_authenticated():
            return []
        
        # Parse relative and absolute dates
        date_obj = self._parse_date_expression(date_str)
        if not date_obj:
            logger.error(f"Invalid date format or expression: {date_str}")
            return []
            
        logger.info(f"Fetching incidents for date: {date_obj.strftime('%Y-%m-%d')}")
        
        # Create qualified query
        start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000") 
        
        query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
        
        # Add status filter if provided
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        # Add owner group filter if provided
        if owner_group:
            query_parts.append(f"'Owner Group'=\"{owner_group}\"")
            
        return self.get_incidents_by_filter(query_parts)
    
    def get_incidents_by_status(self, status, limit=50):
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
        return self.get_incidents_by_filter([f"'Status'=\"{status}\""], limit)
    
    def get_incidents_by_text(self, search_text, fields=None, limit=50):
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
            
        query_parts = []
        for field in fields:
            query_parts.append(f"'{field}' LIKE \"%{search_text}%\"")
            
        # Join conditions with OR for broader search
        query = " OR ".join(query_parts)
        logger.info(f"Searching for text '{search_text}' in fields: {fields}")
        
        return self.get_incidents_by_filter([query], limit)
    
    def get_incidents_by_assignee(self, assignee, status=None, limit=50):
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
        
        query_parts = [f"'Assignee'=\"{assignee}\""]
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        return self.get_incidents_by_filter(query_parts, limit)
    
    def get_incidents_by_owner_group(self, owner_group, status=None, limit=50):
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
        
        query_parts = [f"'Owner Group'=\"{owner_group}\""]
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        return self.get_incidents_by_filter(query_parts, limit)
    
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
        
        # Query parameters - more comprehensive fields
        params = {
            "q": qualified_query,
            "fields": "History Date Time,Action,Description,Status,Changed By,Assigned Group,Details,Previous Value,New Value"
        }
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully retrieved history for incident {incident_id} with {len(result.get('entries', []))} entries")
                return result.get("entries", [])
            else:
                logger.error(f"Get history failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
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

    def _parse_date_expression(self, date_str):
        """
        Parse a date expression into a datetime object.
        Supports formats:
        - 'today', 'yesterday'
        - 'N days ago'
        - 'YYYY-MM-DD'
        - 'MM/DD/YYYY' 
        - 'last week', 'last month'
        
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
            return today - timedelta(days=7)
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


class GeminiClient:
    """
    Enhanced client for Vertex AI Gemini model interactions,
    specialized for handling Remedy incident queries.
    """
    def __init__(self, project_id=PROJECT_ID, region=REGION, model_name=MODEL_NAME):
        """
        Initialize the Gemini client.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            model_name: Gemini model name to use
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.region)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini client for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.model = None
    
    def generate_response(self, prompt, system_instruction=None, temperature=0.2):
        """
        Generate a response using the Gemini model.
        
        Args:
            prompt: The prompt text
            system_instruction: Optional system instruction for the model
            temperature: Temperature parameter (0-1) controlling randomness
            
        Returns:
            str: The generated response
        """
        if not self.model:
            return "Error: Gemini client not properly initialized."
        
        logger.info(f"Generating response for prompt: {prompt[:100]}...")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
            
            # Create content with system instruction if provided
            contents = []
            if system_instruction:
                contents.append(Content(role="user", parts=[system_instruction]))
            contents.append(Content(role="user", parts=[prompt]))
            
            # Generate response
            response = self.model.generate_content(
                contents=contents,
                generation_config=generation_config,
            )
            
            if response and hasattr(response, 'text'):
                logger.info(f"Successfully generated response ({len(response.text)} chars)")
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return "I couldn't generate a response for that query."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"


class RemedyChatbot:
    """
    Integration of Remedy and Gemini clients to create a chatbot system
    for querying and analyzing Remedy incidents.
    """
    def __init__(self, remedy_username=None, remedy_password=None):
        """
        Initialize the chatbot with Remedy and Gemini clients.
        
        Args:
            remedy_username: Optional username for Remedy authentication
            remedy_password: Optional password for Remedy authentication
        """
        self.remedy = RemedyClient(username=remedy_username, password=remedy_password, ssl_verify=False)
        self.gemini = GeminiClient()
        self.conversation_history = []
        self.max_history_length = 10  # Number of exchanges to remember
        
        # Initialize system context (will be updated during operation)
        self.system_context = {
            "incident_counts": {
                "open": 0,
                "closed": 0,
                "in_progress": 0
            },
            "recent_incidents": [],
            "last_query_time": None
        }
        
        logger.info("Remedy Chatbot initialized")
    
    def update_system_context(self):
        """Update the system context with current statistics"""
        try:
            # Get counts of incidents by status
            open_incidents = self.remedy.get_incidents_by_status("Open", limit=1)
            in_progress_incidents = self.remedy.get_incidents_by_status("In Progress", limit=1)
            closed_incidents = self.remedy.get_incidents_by_status("Closed", limit=1)
            
            self.system_context["incident_counts"]["open"] = len(open_incidents)
            self.system_context["incident_counts"]["in_progress"] = len(in_progress_incidents)
            self.system_context["incident_counts"]["closed"] = len(closed_incidents)
            
            # Update recent incidents
            recent_date = datetime.now() - timedelta(days=1)
            recent_incidents = self.remedy.get_incidents_by_date(recent_date.strftime("%Y-%m-%d"))
            self.system_context["recent_incidents"] = recent_incidents[:5]  # Keep only 5 most recent
            
            self.system_context["last_query_time"] = datetime.now()
            
            logger.info("System context updated")
        except Exception as e:
            logger.error(f"Error updating system context: {str(e)}")
    
    def _build_system_prompt(self):
        """
        Build the system prompt for Gemini with detailed instructions.
        
        Returns:
            str: The system prompt with instructions
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        system_prompt = f"""
        You are an intelligent assistant specialized in querying and analyzing BMC Remedy incidents. 
        Your name is "Remedy Assistant" and you have access to the Remedy ITSM system.
        
        Current time: {current_time}
        
        CAPABILITIES:
        - Query incidents by various attributes (ID, status, date, assignee, owner group, text content)
        - Analyze incident data and provide insights
        - Answer questions about Remedy processes and data
        - Provide helpful, accurate responses based on the incident data
        - Extract and explain information from incident descriptions, including tables and technical content
        
        GUIDELINES:
        1. Be concise but thorough in your responses.
        2. Format your responses appropriately based on the content:
           - Use tables for comparing multiple incidents
           - Use bullet points for lists
           - Use paragraphs for explanations
        3. If you don't have enough information, ask clarifying questions.
        4. Always cite the incident numbers when referring to specific incidents.
        5. If the query is ambiguous, acknowledge the ambiguity and provide the most likely interpretation.
        6. If the query is about topics unrelated to Remedy or incidents, respond that you're focused on 
           helping with Remedy incident management but can still try to provide general information.
        7. When displaying incident data, prioritize important fields like Incident Number, Summary, Status, 
           and Assignee, rather than showing all fields.
        8. When referring to sources, mention the specific incident numbers or queries used to gather the information.
        
        You should respond as a helpful, knowledgeable, and professional IT support specialist.
        """
        
        return system_prompt
    
    def _extract_query_intent(self, user_query):
        """
        Extract the intent and parameters from the user's query.
        
        Args:
            user_query: The user's question/command
            
        Returns:
            dict: Intent and parameters
        """
        # Use Gemini to understand the intent
        intent_prompt = f"""
        Analyze the following query about BMC Remedy incidents and extract the intent and parameters.
        Respond in JSON format only with the following structure:
        
        {{
            "intent": "one of [get_incident, search_incidents, query_status, query_date, query_assignee, query_group, general_question, other]",
            "parameters": {{
                "incident_id": "extracted incident ID if present",
                "status": "status mentioned (open, closed, etc.)",
                "date": "date mentioned (today, yesterday, 2023-04-10, etc.)",
                "assignee": "assignee name if mentioned",
                "owner_group": "owner group if mentioned",
                "search_text": "search terms if this is a text search",
                "limit": "number of results requested or default to 5"
            }},
            "query_type": "specific or general"
        }}
        
        For query_type, use "specific" if the query is about a specific incident or set of incidents with clear parameters, 
        and "general" if it's a more general question about Remedy or processes.
        
        The query is: {user_query}
        """
        
        try:
            response = self.gemini.generate_response(intent_prompt, temperature=0.1)
            # Parse the JSON response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            intent_data = json.loads(response)
            logger.info(f"Extracted intent: {intent_data['intent']}")
            return intent_data
        except Exception as e:
            logger.error(f"Error extracting query intent: {str(e)}")
            # Return a default/fallback intent
            return {
                "intent": "other",
                "parameters": {},
                "query_type": "general"
            }
    
    def _fetch_data_for_intent(self, intent_data):
        """
        Fetch the relevant data from Remedy based on the intent.
        
        Args:
            intent_data: Dict with intent and parameters
            
        Returns:
            dict: Retrieved data and metadata
        """
        intent = intent_data["intent"]
        params = intent_data["parameters"]
        
        # Default limit
        limit = int(params.get("limit", 5))
        
        result = {
            "intent": intent,
            "data": None,
            "metadata": {
                "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result_count": 0,
                "query_details": params
            }
        }
        
        try:
            # Handle different intents
            if intent == "get_incident":
                incident_id = params.get("incident_id")
                if incident_id:
                    data = self.remedy.get_incident(incident_id)
                    if data:
                        # Add history data for enrichment
                        history = self.remedy.get_incident_history(incident_id)
                        result["data"] = {"incident": data, "history": history}
                        result["metadata"]["result_count"] = 1
                        result["metadata"]["success"] = True
                    else:
                        result["metadata"]["success"] = False
                        result["metadata"]["error"] = f"Incident {incident_id} not found"
                else:
                    result["metadata"]["success"] = False
                    result["metadata"]["error"] = "No incident ID provided"
            
            elif intent == "search_incidents":
                search_text = params.get("search_text")
                if search_text:
                    data = self.remedy.get_incidents_by_text(search_text, limit=limit)
                    result["data"] = data
                    result["metadata"]["result_count"] = len(data)
                    result["metadata"]["success"] = True
                else:
                    result["metadata"]["success"] = False
                    result["metadata"]["error"] = "No search text provided"
            
            elif intent == "query_status":
                status = params.get("status")
                if status:
                    data = self.remedy.get_incidents_by_status(status, limit=limit)
                    result["data"] = data
                    result["metadata"]["result_count"] = len(data)
                    result["metadata"]["success"] = True
                else:
                    result["metadata"]["success"] = False
                    result["metadata"]["error"] = "No status provided"
            
            elif intent == "query_date":
                date_str = params.get("date")
                if date_str:
                    data = self.remedy.get_incidents_by_date(date_str)
                    result["data"] = data
                    result["metadata"]["result_count"] = len(data)
                    result["metadata"]["success"] = True
                else:
                    result["metadata"]["success"] = False
                    result["metadata"]["error"] = "No date provided"
            
            elif intent == "query_assignee":
                assignee = params.get("assignee")
                if assignee:
                    data = self.remedy.get_incidents_by_assignee(assignee, limit=limit)
                    result["data"] = data
                    result["metadata"]["result_count"] = len(data)
                    result["metadata"]["success"] = True
                else:
                    result["metadata"]["success"] = False
                    result["metadata"]["error"] = "No assignee provided"
            
            elif intent == "query_group":
                group = params.get("owner_group")
                if group:
                    data = self.remedy.get_incidents_by_owner_group(group, limit=limit)
                    result["data"] = data
                    result["metadata"]["result_count"] = len(data)
                    result["metadata"]["success"] = True
                else:
                    result["metadata"]["success"] = False
                    result["metadata"]["error"] = "No owner group provided"
            
            else:
                # For general questions, we don't need to fetch specific data
                result["metadata"]["success"] = True
                result["data"] = []
        
        except Exception as e:
            logger.error(f"Error fetching data for intent {intent}: {str(e)}")
            result["metadata"]["success"] = False
            result["metadata"]["error"] = str(e)
        
        return result
    
    def _format_incidents_for_prompt(self, incidents, detailed=False):
        """
        Format incident data for inclusion in a prompt.
        
        Args:
            incidents: List of incident data dictionaries
            detailed: Whether to include detailed information
            
        Returns:
            str: Formatted incident data
        """
        if not incidents:
            return "No incidents found."
            
        result = []
        
        for i, incident in enumerate(incidents, 1):
            if not isinstance(incident, dict) or "values" not in incident:
                result.append(f"Incident {i}: Invalid data format")
                continue
                
            values = incident.get("values", {})
            
            # Basic format for all incidents
            incident_info = [
                f"--- Incident {i} ---",
                f"Incident Number: {values.get('Incident Number', 'N/A')}",
                f"Summary: {values.get('Summary', 'N/A')}",
                f"Status: {values.get('Status', 'N/A')}",
                f"Priority: {values.get('Priority', 'N/A')}",
                f"Assignee: {values.get('Assignee', 'N/A')}",
                f"Owner Group: {values.get('Owner Group', 'N/A')}",
                f"Submit Date: {values.get('Submit Date', 'N/A')}"
            ]
            
            # Add more details if requested
            if detailed:
                incident_info.extend([
                    f"Description: {values.get('Description', 'N/A')}",
                    f"Impact: {values.get('Impact', 'N/A')}",
                    f"Environment: {values.get('Environment', 'N/A')}",
                    f"Last Modified: {values.get('Last Modified Date', 'N/A')}",
                    f"Resolution: {values.get('Resolution', 'N/A')}"
                ])
            
            result.append("\n".join(incident_info))
        
        return "\n\n".join(result)
    
    def _format_incident_history(self, history):
        """
        Format incident history data for inclusion in a prompt.
        
        Args:
            history: List of history entries
            
        Returns:
            str: Formatted history data
        """
        if not history:
            return "No history entries found."
            
        result = ["--- Incident History ---"]
        
        for entry in history:
            if not isinstance(entry, dict) or "values" not in entry:
                continue
                
            values = entry.get("values", {})
            
            entry_info = [
                f"Date/Time: {values.get('History Date Time', 'N/A')}",
                f"Action: {values.get('Action', 'N/A')}",
                f"Changed By: {values.get('Changed By', 'N/A')}",
                f"Description: {values.get('Description', 'N/A')}"
            ]
            
            # Add previous and new values if present
            if "Previous Value" in values and "New Value" in values:
                entry_info.append(f"Changed from '{values.get('Previous Value')}' to '{values.get('New Value')}'")
            
            result.append(" | ".join(entry_info))
        
        return "\n".join(result)
    
    def _build_response_prompt(self, user_query, query_result):
        """
        Build the prompt for Gemini to generate a response based on query results.
        
        Args:
            user_query: Original user query
            query_result: Result of data fetching
            
        Returns:
            str: Prompt for generating the response
        """
        intent = query_result["intent"]
        data = query_result["data"]
        metadata = query_result["metadata"]
        
        # Start with the user's query
        prompt_parts = [
            f"USER QUERY: {user_query}",
            "",
            f"QUERY INTENT: {intent}",
            f"QUERY PARAMETERS: {json.dumps(metadata['query_details'])}",
            f"QUERY TIME: {metadata['query_time']}",
            f"RESULTS FOUND: {metadata['result_count']}",
            ""
        ]
        
        # Add data based on intent and success
        if metadata["success"]:
            if intent == "get_incident" and data:
                # Single incident with history
                incident = data.get("incident", {})
                history = data.get("history", [])
                
                if isinstance(incident, dict) and "values" in incident:
                    values = incident.get("values", {})
                    prompt_parts.append("INCIDENT DATA:")
                    
                    # Add all incident fields that are present
                    for key, value in values.items():
                        prompt_parts.append(f"{key}: {value}")
                    
                    # Add history if available
                    if history:
                        prompt_parts.append("\nINCIDENT HISTORY:")
                        prompt_parts.append(self._format_incident_history(history))
                else:
                    prompt_parts.append("INCIDENT DATA: Invalid data format")
            
            elif intent in ["search_incidents", "query_status", "query_date", "query_assignee", "query_group"]:
                # Multiple incidents
                prompt_parts.append("INCIDENTS FOUND:")
                
                if not data:
                    prompt_parts.append("No incidents found matching the criteria.")
                else:
                    # Format more detailed for fewer results
                    detailed = len(data) <= 3
                    prompt_parts.append(self._format_incidents_for_prompt(data, detailed))
                    
                    # Add a summary for larger result sets
                    if len(data) > 3:
                        status_counts = {}
                        priority_counts = {}
                        
                        for incident in data:
                            values = incident.get("values", {})
                            status = values.get("Status", "Unknown")
                            priority = values.get("Priority", "Unknown")
                            
                            status_counts[status] = status_counts.get(status, 0) + 1
                            priority_counts[priority] = priority_counts.get(priority, 0) + 1
                        
                        prompt_parts.append("\nSUMMARY STATISTICS:")
                        prompt_parts.append(f"Total incidents: {len(data)}")
                        prompt_parts.append(f"Status distribution: {json.dumps(status_counts)}")
                        prompt_parts.append(f"Priority distribution: {json.dumps(priority_counts)}")
        else:
            # Query failed
            prompt_parts.append(f"ERROR: {metadata.get('error', 'Unknown error')}")
        
        # Instructions for response generation
        prompt_parts.append("\n" + "-" * 50 + "\n")
        prompt_parts.append("INSTRUCTIONS FOR RESPONSE:")
        prompt_parts.append("1. Provide a clear, concise response to the user's query based on the data above.")
        prompt_parts.append("2. Format your response appropriately (tables for multiple incidents, paragraphs for explanations, etc.)")
        prompt_parts.append("3. If the query failed, explain the issue and suggest alternatives.")
        prompt_parts.append("4. If data is available, analyze it to provide insights beyond just listing the data.")
        prompt_parts.append("5. Cite specific incident numbers when referencing incidents.")
        prompt_parts.append("6. If the results are limited, mention that there may be more incidents matching the criteria.")
        
        if metadata["result_count"] == 0 and metadata["success"]:
            prompt_parts.append("7. Suggest alternative queries since no results were found.")
        
        return "\n".join(prompt_parts)
    
    def process_query(self, user_query):
        """
        Process a user query and generate a response.
        
        Args:
            user_query: The user's question/command
            
        Returns:
            str: The response to the user
        """
        # Ensure we're authenticated with Remedy
        if not self.remedy.ensure_authenticated():
            return "I'm having trouble connecting to the Remedy system. Please check your credentials and try again."
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "text": user_query})
        
        try:
            # 1. Extract intent from the query
            intent_data = self._extract_query_intent(user_query)
            
            # 2. Fetch relevant data from Remedy
            query_result = self._fetch_data_for_intent(intent_data)
            
            # 3. Build the prompt for response generation
            response_prompt = self._build_response_prompt(user_query, query_result)
            
            # 4. Generate response using Gemini
            system_prompt = self._build_system_prompt()
            response = self.gemini.generate_response(response_prompt, system_prompt)
            
            # 5. Update conversation history
            self.conversation_history.append({"role": "assistant", "text": response})
            if len(self.conversation_history) > self.max_history_length * 2:
                # Keep only the last N exchanges
                self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    def run_chat_loop(self):
        """
        Run an interactive chat loop for the chatbot.
        """
        print("\n" + "=" * 50)
        print("Welcome to Remedy Assistant!")
        print("Ask me questions about incidents in your Remedy system.")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 50 + "\n")
        
        # Ensure we're authenticated at the start
        if not self.remedy.ensure_authenticated():
            print("Failed to authenticate with Remedy. Please check your credentials.")
            return
            
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using Remedy Assistant. Goodbye!")
                break
                
            print("\nProcessing your query...")
            response = self.process_query(user_input)
            
            print("\nAssistant:", end=" ")
            # Display the response with proper formatting
            for line in response.split('\n'):
                print(textwrap.fill(line, width=100) if len(line) > 100 else line)


if __name__ == "__main__":
    # You can either provide credentials here or let the program prompt for them
    remedy_username = os.environ.get("REMEDY_USERNAME")
    remedy_password = os.environ.get("REMEDY_PASSWORD")
    
    chatbot = RemedyChatbot(remedy_username, remedy_password)
    chatbot.run_chat_loop()




























#!/usr/bin/env python3
import logging
import os
import sys
import re
import json
import requests
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
import ssl
import urllib3

# Disable SSL verification globally as requested
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_gemini_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraGeminiAssistant")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://your-jira-instance.com")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME", "")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")

class JiraClient:
    """Class for interacting with Jira API."""
    
    def __init__(self, base_url, username, token):
        """Initialize the Jira client."""
        self.base_url = base_url
        self.auth = (username, token)
        self.headers = {"Content-Type": "application/json"}
        self.verify = False  # Disable SSL verification
    
    def test_connection(self):
        """Test connection to Jira."""
        try:
            logger.info("Testing connection to Jira...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            server_info = response.json()
            logger.info(f"Connection to Jira successful! Server version: {server_info.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {str(e)}")
            error_details = {
                "error": str(e),
                "type": str(type(e)),
                "url": f"{self.base_url}/rest/api/2/serverInfo"
            }
            logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
            return False
    
    def get_issue(self, issue_key):
        """Get a specific issue by its key."""
        try:
            logger.info(f"Fetching issue: {issue_key}")
            params = {
                "expand": "renderedFields,names,schema,transitions,operations,editmeta,changelog,attachment"
            }
            response = requests.get(
                f"{self.base_url}/rest/api/2/issue/{issue_key}",
                params=params,
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            return None
    
    def search_issues(self, jql, start_at=0, max_results=50, fields=None, expand=None):
        """Search for issues using JQL (Jira Query Language)."""
        try:
            logger.info(f"Searching issues with JQL: {jql}")
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results
            }
            
            if fields:
                params["fields"] = ",".join(fields) if isinstance(fields, list) else fields
                
            if expand:
                params["expand"] = expand
                
            response = requests.get(
                f"{self.base_url}/rest/api/2/search",
                params=params,
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            search_results = response.json()
            total = search_results.get("total", 0)
            logger.info(f"Search returned {total} issues")
            return search_results
        except Exception as e:
            logger.error(f"Failed to search issues: {str(e)}")
            return None
    
    def get_all_issues(self, jql, fields=None, max_results=1000):
        """Get all issues matching a JQL query, handling pagination."""
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        all_issues = []
        start_at = 0
        page_size = 100  # Jira recommends 100 for optimal performance
        
        while True:
            search_results = self.search_issues(
                jql=jql,
                start_at=start_at,
                max_results=page_size,
                fields=fields
            )
            
            if not search_results or not search_results.get("issues"):
                break
                
            issues = search_results.get("issues", [])
            all_issues.extend(issues)
            
            # Check if we've reached the total or our max limit
            total = search_results.get("total", 0)
            if len(all_issues) >= total or len(all_issues) >= max_results:
                break
                
            # Move to next page
            start_at += len(issues)
            
            # If no issues were returned, we're done
            if len(issues) == 0:
                break
        
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    def get_issue_content(self, issue_key):
        """Get the content of an issue in a format suitable for the assistant."""
        issue = self.get_issue(issue_key)
        if not issue:
            return None
        
        # Extract key metadata
        metadata = {
            "key": issue.get("key"),
            "summary": issue["fields"].get("summary"),
            "type": issue["fields"].get("issuetype", {}).get("name"),
            "status": issue["fields"].get("status", {}).get("name"),
            "created": issue["fields"].get("created"),
            "updated": issue["fields"].get("updated"),
            "priority": issue["fields"].get("priority", {}).get("name") if issue["fields"].get("priority") else None,
            "labels": issue["fields"].get("labels", []),
            "resolution": issue["fields"].get("resolution", {}).get("name") if issue["fields"].get("resolution") else None,
            "url": f"{self.base_url}/browse/{issue.get('key')}"
        }
        
        # Extract people
        if issue["fields"].get("assignee"):
            metadata["assignee"] = issue["fields"].get("assignee", {}).get("displayName")
            
        if issue["fields"].get("reporter"):
            metadata["reporter"] = issue["fields"].get("reporter", {}).get("displayName")
        
        # Extract content fields
        content_parts = []
        
        # Add summary
        summary = issue["fields"].get("summary", "")
        if summary:
            content_parts.append(f"Summary: {summary}")
        
        # Add description - try to use rendered HTML if available
        if "renderedFields" in issue and issue["renderedFields"].get("description"):
            description_html = issue["renderedFields"].get("description")
            # Basic HTML tag stripping
            description_text = re.sub(r'<[^>]+>', ' ', description_html)
            description_text = re.sub(r'\s+', ' ', description_text).strip()
            content_parts.append(f"Description: {description_text}")
        elif issue["fields"].get("description"):
            description = issue["fields"].get("description")
            if isinstance(description, dict):
                # Handle Atlassian Document Format
                desc_text = self._extract_text_from_adf(description)
                content_parts.append(f"Description: {desc_text}")
            else:
                content_parts.append(f"Description: {description}")
        
        # Add attachments info
        if issue["fields"].get("attachment"):
            attachments = issue["fields"].get("attachment", [])
            if attachments:
                attachment_info = []
                for attachment in attachments:
                    attachment_info.append(f"{attachment.get('filename')} ({attachment.get('mimeType')})")
                content_parts.append(f"Attachments: {', '.join(attachment_info)}")
        
        # Add comments - try to use rendered content if available
        if "renderedFields" in issue and issue["renderedFields"].get("comment", {}).get("comments"):
            comments = issue["renderedFields"].get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "unknown")
                created = comment.get("created", "")
                
                # Extract text from HTML
                comment_html = comment.get("body", "")
                comment_text = re.sub(r'<[^>]+>', ' ', comment_html)
                comment_text = re.sub(r'\s+', ' ', comment_text).strip()
                
                content_parts.append(f"Comment by {author} on {created}: {comment_text}")
        elif issue["fields"].get("comment", {}).get("comments"):
            comments = issue["fields"].get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "unknown")
                created = comment.get("created", "")
                
                comment_body = comment.get("body")
                if isinstance(comment_body, dict):
                    # Handle Atlassian Document Format
                    comment_text = self._extract_text_from_adf(comment_body)
                    content_parts.append(f"Comment by {author} on {created}: {comment_text}")
                else:
                    content_parts.append(f"Comment by {author} on {created}: {comment_body}")
        
        # Add custom fields that might contain important information
        for field_id, field_value in issue["fields"].items():
            # Skip fields we've already processed and empty values
            if field_id in ["summary", "description", "comment", "attachment", "assignee", "reporter", 
                           "issuetype", "status", "created", "updated", "priority", "labels", "resolution"]:
                continue
                
            if not field_value:
                continue
                
            # Get field name if available
            field_name = field_id
            if "names" in issue and field_id in issue["names"]:
                field_name = issue["names"][field_id]
                
            # Handle different value types
            if isinstance(field_value, dict) and "value" in field_value:
                content_parts.append(f"{field_name}: {field_value['value']}")
            elif isinstance(field_value, dict) and "name" in field_value:
                content_parts.append(f"{field_name}: {field_value['name']}")
            elif isinstance(field_value, list) and all(isinstance(item, dict) for item in field_value):
                values = []
                for item in field_value:
                    if "value" in item:
                        values.append(item["value"])
                    elif "name" in item:
                        values.append(item["name"])
                if values:
                    content_parts.append(f"{field_name}: {', '.join(values)}")
            elif not isinstance(field_value, (dict, list)):
                content_parts.append(f"{field_name}: {field_value}")
        
        # Combine all content
        full_content = "\n\n".join(content_parts)
        
        # Return formatted content with metadata
        return {
            "metadata": metadata,
            "content": full_content
        }
    
    def _extract_text_from_adf(self, adf_doc):
        """Extract plain text from Atlassian Document Format (ADF)."""
        if not adf_doc or not isinstance(adf_doc, dict):
            return ""
        
        text_parts = []
        
        def extract_from_content(content_list):
            parts = []
            if not content_list or not isinstance(content_list, list):
                return ""
                
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                    
                # Extract text node
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                
                # Handle links
                if item.get("type") == "link" and "attrs" in item and "href" in item.get("attrs", {}):
                    href = item.get("attrs", {}).get("href", "")
                    link_text = extract_from_content(item.get("content", []))
                    parts.append(f"{link_text} ({href})")
                
                # Handle mentions
                if item.get("type") == "mention" and "attrs" in item and "text" in item.get("attrs", {}):
                    mention_text = item.get("attrs", {}).get("text", "")
                    parts.append(f"@{mention_text}")
                
                # Extract code blocks
                if item.get("type") == "codeBlock":
                    code_text = extract_from_content(item.get("content", []))
                    parts.append(f"Code: {code_text}")
                
                # Extract from content recursively
                if "content" in item and isinstance(item["content"], list):
                    parts.append(extract_from_content(item["content"]))
                    
            return " ".join(parts)
        
        # Extract from main content array
        for item in adf_doc.get("content", []):
            text_parts.append(extract_from_content([item]))
        
        return " ".join(text_parts)
        
    def get_issue_type_metadata(self):
        """Get issue type metadata for better understanding of the Jira instance's structure."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/2/issuetype",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get issue type metadata: {str(e)}")
            return None
            
    def get_field_metadata(self):
        """Get field metadata for better understanding of custom fields."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/2/field",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get field metadata: {str(e)}")
            return None

class JiraGeminiAssistant:
    """Enhanced assistant using Gemini and Jira integration."""
    
    def __init__(self, jira_client):
        """Initialize the assistant."""
        # Initialize Vertex AI with disabled SSL verification
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        self.jira_client = jira_client
        self.conversation_history = []
        
        # Load system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Cache for issue metadata to reduce API calls
        self.issue_cache = {}
        
        # Get Jira metadata for better understanding
        self.issue_types = self.jira_client.get_issue_type_metadata()
        self.field_metadata = self.jira_client.get_field_metadata()
        
    def _create_system_prompt(self):
        """Create an enhanced system prompt for the Gemini model."""
        return """
        # JIRA ASSISTANT SYSTEM INSTRUCTIONS

        You are JiraGenius, an advanced assistant designed to help users with their Jira environment. You provide accurate, comprehensive, and helpful responses to all Jira-related questions while maintaining a professional yet friendly tone.

        ## CORE CAPABILITIES
        1. **Jira Expertise**: You can answer questions about Jira tickets, projects, workflows, users, and all Jira-related functionality.
        2. **Data Analysis**: You can analyze and interpret information from Jira tickets, including text, images, tables, and attachments.
        3. **Search Functions**: You can find and summarize tickets based on various criteria (project, status, assignee, etc.).
        4. **Ticket Details**: You can provide comprehensive information about specific tickets.
        5. **Visual Understanding**: You can interpret and describe images, charts, and tables within Jira tickets.
        6. **Advanced Formatting**: You format responses optimally based on the content type (tables, lists, paragraphs).
        7. **Proactive Clarification**: You ask follow-up questions when needed to provide the most accurate response.
        8. **Technical Context**: You understand software development, project management, and IT terminology.
        9. **Citation and Links**: You provide links to relevant Jira tickets and pages in your responses.
        10. **Instant Responses**: You prioritize providing quick, concise answers when appropriate.

        ## RESPONSE GUIDELINES

        ### Format and Structure
        - Use appropriate formatting for different content types (tables for comparisons, bullet points for lists, etc.)
        - Structure complex responses with clear headings and sections
        - Highlight key information visually when appropriate
        - For ticket details, always begin with a summary card showing key information
        - For search results, use tables with columns for Key, Summary, Status, and Assignee

        ### Content Quality
        - Be accurate and factual above all else
        - Be comprehensive but concise - cover all aspects without unnecessary verbosity
        - Provide context and background information when beneficial
        - Link to specific tickets whenever mentioned using the full URL
        - Include relevant metadata (created date, status changes, assignee history) when discussing tickets
        - When describing images or visual elements, be detailed and explain their relevance to the ticket

        ### Tone and Style
        - Maintain a professional but conversational tone
        - Use technical terminology appropriately for the context and user expertise level
        - Be helpful and solution-oriented
        - Show empathy for user challenges
        - Be confident in assertions but acknowledge limitations when appropriate
        - Use a friendly, approachable writing style that builds rapport
        
        ### Follow-up Questions
        - Ask clarifying questions when user queries are ambiguous
        - Frame questions to narrow down exactly what the user needs
        - Limit follow-up questions to one per response
        - Make follow-up questions specific and directly relevant to improving your answer
        - When a query could have multiple interpretations, ask for clarification rather than assuming

        ## RESPONSE STRATEGIES BY QUERY TYPE

        ### For Ticket Search Queries
        - Return results in a table format with columns for Key, Summary, Status, and Assignee
        - Include the total number of matching tickets
        - Provide direct links to each ticket
        - Sort results by priority or recency unless another order is specified
        - Offer pagination information if there are many results
        - Include a brief summary of the search criteria used

        ### For Specific Ticket Details
        - Begin with a "Ticket Card" showing Key, Summary, Status, Priority, Assignee, and Creation Date
        - Include a direct link to the ticket
        - Organize information in logical sections (Description, Comments, History, Attachments, etc.)
        - Highlight important updates or changes
        - Include all relevant metadata (components, labels, etc.)
        - Summarize long descriptions or comments while preserving key details
        - For tickets with many comments, focus on the most recent or most relevant

        ### For Project Overviews
        - Provide key statistics (open tickets, recently completed, upcoming)
        - Summarize the project's current status and key milestones
        - Highlight any blockers or critical issues
        - List key contributors and their roles
        - Include links to important project resources
        - Mention recent activity and upcoming deadlines

        ### For Technical Questions
        - Break down complex issues step by step
        - Differentiate between verified solutions and theoretical approaches
        - Include practical examples where relevant
        - Reference documentation when appropriate
        - Provide context for technical terminology
        - Include both immediate fixes and long-term solutions when applicable

        ### For Irrelevant Questions
        - If the question is completely unrelated to Jira but is a general knowledge question you can answer, provide a brief, helpful response
        - If the question is inappropriate or outside your capabilities, politely redirect to Jira-related topics
        - Always maintain a professional tone even when declining to answer
        - Suggest related Jira topics that might be more helpful

        ## SPECIAL HANDLING

        ### Images and Attachments
        - When discussing images in tickets, describe what you can see and how it relates to the ticket
        - Identify charts, diagrams, screenshots, and explain their content
        - For technical screenshots, identify the application or system shown
        - For error messages in images, transcribe them when possible

        ### Data Tables
        - Maintain table structure in your response
        - Summarize key patterns or insights from the table
        - Highlight anomalies or important data points
        - For large tables, focus on the most relevant sections

        ### User References
        - When mentioning users, include their role and responsibilities if known
        - Respect privacy by not revealing sensitive personal information
        - Focus on work activities and contributions rather than personal attributes
        - Use formal names and titles when appropriate

        ### Workflows and Processes
        - Explain the current stage in the workflow and next steps
        - Identify bottlenecks or blockers in the process
        - Suggest workflow improvements when appropriate
        - Reference established processes and best practices

        Remember, your primary goal is to provide immediate, accurate, and helpful information about the user's Jira environment in a professional, friendly manner. Always include relevant links, format your response appropriately for the content, and be proactive in asking clarifying questions when needed.
        """
    
    def generate_response(self, user_query):
        """Generate a response to a user query."""
        logger.info(f"Generating response for: {user_query}")
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Check if query is about a specific Jira ticket
        ticket_pattern = r'\b[A-Z]+-\d+\b'  # e.g., PROJ-123
        ticket_matches = re.findall(ticket_pattern, user_query)
        
        # Enhance the prompt with relevant Jira data if applicable
        enhanced_prompt = self._enhance_prompt_with_jira_data(user_query, ticket_matches)
        
        try:
            # Configure generation parameters for faster, high-quality responses
            generation_config = GenerationConfig(
                temperature=0.2,  # Lower temperature for more factual responses
                top_p=0.95,
                max_output_tokens=8192,
            )
            
            # Generate response - streaming for speed
            logger.info("Generating response...")
            response_text = ""
            
            full_prompt = f"""
            {self.system_prompt}
            
            # CONVERSATION HISTORY
            {self._format_conversation_history()}
            
            # JIRA CONTEXT INFORMATION
            {enhanced_prompt}
            
            # USER QUERY
            {user_query}
            
            # YOUR RESPONSE
            """
            
            for chunk in self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    response_text += chunk.candidates[0].text
                    
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Cap conversation history to prevent context overflow
            if len(self.conversation_history) > 10:  # Keep last 5 exchanges (10 messages)
                self.conversation_history = self.conversation_history[-10:]
                
            logger.info(f"Response length: {len(response_text)} characters")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response. Please try again or contact support if the issue persists."
    
    def _format_conversation_history(self):
        """Format conversation history for the prompt."""
        formatted_history = ""
        for message in self.conversation_history[-8:]:  # Use last 8 messages max
            role = message["role"].capitalize()
            content = message["content"]
            formatted_history += f"{role}: {content}\n\n"
        return formatted_history
    
    def _enhance_prompt_with_jira_data(self, user_query, ticket_matches):
        """Enhance the prompt with relevant Jira data."""
        enhanced_data = []
        
        # If specific tickets are mentioned, fetch their details
        if ticket_matches:
            for ticket_id in ticket_matches:
                # Check cache first
                if ticket_id in self.issue_cache:
                    ticket_content = self.issue_cache[ticket_id]
                else:
                    ticket_content = self.jira_client.get_issue_content(ticket_id)
                    if ticket_content:
                        self.issue_cache[ticket_id] = ticket_content
                
                if ticket_content:
                    enhanced_data.append(f"## JIRA TICKET: {ticket_id}")
                    enhanced_data.append(f"URL: {ticket_content['metadata']['url']}")
                    enhanced_data.append(f"Summary: {ticket_content['metadata']['summary']}")
                    enhanced_data.append(f"Type: {ticket_content['metadata']['type']}")
                    enhanced_data.append(f"Status: {ticket_content['metadata']['status']}")
                    enhanced_data.append(f"Created: {ticket_content['metadata']['created']}")
                    enhanced_data.append(f"Updated: {ticket_content['metadata']['updated']}")
                    
                    if 'assignee' in ticket_content['metadata']:
                        enhanced_data.append(f"Assignee: {ticket_content['metadata']['assignee']}")
                    
                    if 'reporter' in ticket_content['metadata']:
                        enhanced_data.append(f"Reporter: {ticket_content['metadata']['reporter']}")
                    
                    if 'priority' in ticket_content['metadata'] and ticket_content['metadata']['priority']:
                        enhanced_data.append(f"Priority: {ticket_content['metadata']['priority']}")
                    
                    if 'labels' in ticket_content['metadata'] and ticket_content['metadata']['labels']:
                        enhanced_data.append(f"Labels: {', '.join(ticket_content['metadata']['labels'])}")
                    
                    enhanced_data.append("Content:")
                    enhanced_data.append(ticket_content['content'])
        
        # If the query seems to be a search rather than about specific tickets
        elif any(term in user_query.lower() for term in ['search', 'find', 'look for', 'show me', 'list', 'get', 'what are', 'tickets']):
            # Try to extract potential JQL terms from the query
            potential_jql = self._extract_jql_from_query(user_query)
            if potential_jql:
                try:
                    issues = self.jira_client.search_issues(potential_jql, max_results=15)
                    if issues and issues.get('issues'):
                        enhanced_data.append(f"## SEARCH RESULTS")
                        enhanced_data.append(f"JQL Query: {potential_jql}")
                        enhanced_data.append(f"Total matches: {issues.get('total', 0)}")
                        
                        for i, issue in enumerate(issues.get('issues', [])[:15]):
                            issue_key = issue.get('key')
                            summary = issue.get('fields', {}).get('summary', 'No summary')
                            status = issue.get('fields', {}).get('status', {}).get('name', 'Unknown')
                            assignee = issue.get('fields', {}).get('assignee', {}).get('displayName', 'Unassigned')
                            
                            enhanced_data.append(f"{i+1}. {issue_key}: {summary}")
                            enhanced_data.append(f"   Status: {status} | Assignee: {assignee}")
                            enhanced_data.append(f"   Link: {self.jira_client.base_url}/browse/{issue_key}")
                except Exception as e:
                    logger.error(f"Error executing JQL search: {str(e)}")
                    enhanced_data.append(f"Failed to execute search with JQL: {potential_jql}")
                    enhanced_data.append(f"Error: {str(e)}")
        
        # Return the enhanced prompt
        if enhanced_data:
            return "\n".join(enhanced_data)
        else:
            return "No specific Jira ticket information is available for this query."
    
    def _extract_jql_from_query(self, user_query):
        """
        Extract JQL from a natural language query.
        This implementation parses common search patterns from the query.
        """
        query = user_query.lower()
        jql_parts = []
        
        # Check for project reference
        project_match = re.search(r'(?:in |for |project |projects )(?:called |named |)["\'()]?([a-zA-Z0-9]+)["\'()]?', query)
        if project_match:
            jql_parts.append(f"project = {project_match.group(1).upper()}")
        
        # Check for status
        status_terms = {
            'open': ['open', 'active', 'in progress', 'ongoing', 'not closed', 'not done', 'not resolved'],
            'closed': ['closed', 'resolved', 'done', 'completed', 'finished'],
            'blocked': ['blocked', 'impediment', 'stuck'],
            'in progress': ['in progress', 'working', 'active', 'being worked on'],
            'new': ['new', 'to do', 'backlog', 'not started']
        }
        
        for status, terms in status_terms.items():
            if any(term in query for term in terms):
                if status == 'open':
                    jql_parts.append('status not in (Closed, Resolved, Done)')
                elif status == 'closed':
                    jql_parts.append('status in (Closed, Resolved, Done)')
                else:
                    jql_parts.append(f'status = "{status}"')
                break
        
        # Check for assignee
        assigned_match = re.search(r'(?:assigned to|assignee is|owned by) ["\'()]?([^"\'()]+)["\'()]?', query)
        if assigned_match:
            assignee = assigned_match.group(1).strip()
            if assignee in ['me', 'myself', 'i']:
                jql_parts.append('assignee = currentUser()')
            elif assignee in ['no one', 'nobody', 'not assigned', 'unassigned']:
                jql_parts.append('assignee is EMPTY')
            else:
                jql_parts.append(f'assignee ~ "{assignee}"')
        
        # Check for issue type
        type_terms = ['bug', 'task', 'story', 'epic', 'feature', 'improvement', 'enhancement']
        for term in type_terms:
            if term in query and any(x in query for x in [f"type {term}", f"{term}s", f"{term} tickets"]):
                jql_parts.append(f'issuetype = "{term}"')
                break
        
        # Check for priority
        priority_terms = ['blocker', 'critical', 'major', 'minor', 'trivial', 'high', 'medium', 'low']
        for term in priority_terms:
            if term in query and any(x in query for x in [f"priority {term}", f"{term} priority"]):
                jql_parts.append(f'priority = "{term}"')
                break
        
        # Check for text search
        text_match = re.search(r'(?:containing|with text|about|mentions|related to) ["\'()]?([^"\'()]+)["\'()]?', query)
        if text_match:
            search_text = text_match.group(1).strip()
            jql_parts.append(f'text ~ "{search_text}"')
        
        # Check for recent tickets
        time_terms = {
            'today': 'created >= startOfDay()',
            'yesterday': 'created >= startOfDay(-1d) AND created < startOfDay()',
            'this week': 'created >= startOfWeek()',
            'last week': 'created >= startOfWeek(-1w) AND created < startOfWeek()',
            'this month': 'created >= startOfMonth()',
            'last month': 'created >= startOfMonth(-1M) AND created < startOfMonth()'
        }
        
        for term, jql in time_terms.items():
            if term in query:
                jql_parts.append(jql)
                break
                
        # Check for reporter
        reporter_match = re.search(r'(?:reported by|created by|raised by) ["\'()]?([^"\'()]+)["\'()]?', query)
        if reporter_match:
            reporter = reporter_match.group(1).strip()
            if reporter in ['me', 'myself', 'i']:
                jql_parts.append('reporter = currentUser()')
            else:
                jql_parts.append(f'reporter ~ "{reporter}"')
        
        # If we have any parts, combine them with AND
        if jql_parts:
            jql = " AND ".join(jql_parts)
            # Add sorting if not already specified
            if "ORDER BY" not in jql:
                if "created" in jql:
                    jql += " ORDER BY created DESC"
                else:
                    jql += " ORDER BY updated DESC"
                return jql
        else:
            # If we don't have specific parts but it seems like a search query, 
            # return a default recent issues query
            if any(term in query for term in ['search', 'find', 'list', 'show', 'get', 'recent']):
                return "ORDER BY updated DESC"
            
            return None

    def ask_follow_up_question(self, user_query, response_so_far):
        """Decide if a follow-up question is needed to clarify the query."""
        # Check if the query is clear enough
        prompt = f"""
        Based on the user's query and my response so far, determine if I need to ask a clarifying follow-up question.

        User query: "{user_query}"
        
        My response so far: "{response_so_far}"
        
        Should I ask a clarifying question? If yes, what specific question should I ask to better help the user?
        
        Return your answer in this format:
        NEED_CLARIFICATION: [YES/NO]
        QUESTION: [The specific question to ask, if needed]
        """
        
        try:
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=1024,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            result_text = response.candidates[0].text
            
            # Parse the result
            need_clarification = "NEED_CLARIFICATION: YES" in result_text
            
            if need_clarification:
                question_match = re.search(r'QUESTION: (.*)', result_text)
                if question_match:
                    return question_match.group(1).strip()
            
            return None
        except Exception as e:
            logger.error(f"Error determining follow-up question: {str(e)}")
            return None

class InteractiveJiraAssistant:
    """Interactive command-line interface for JiraGeminiAssistant."""
    
    def __init__(self):
        """Initialize the interactive assistant."""
        # Initialize Jira client
        self.jira_client = self._initialize_jira_client()
        
        # Initialize Gemini assistant
        if self.jira_client:
            self.assistant = JiraGeminiAssistant(self.jira_client)
            logger.info("JiraGeminiAssistant initialized successfully.")
        else:
            logger.error("Failed to initialize JiraGeminiAssistant.")
            sys.exit(1)
    
    def _initialize_jira_client(self):
        """Initialize and test the Jira client connection."""
        logger.info("Initializing Jira client...")
        
        # Get Jira credentials from environment or ask user
        base_url = JIRA_BASE_URL
        username = JIRA_USERNAME
        token = JIRA_TOKEN
        
        if not base_url or base_url == "https://your-jira-instance.com":
            base_url = input("Enter your Jira base URL (e.g., https://your-company.atlassian.net): ")
        
        if not username:
            username = input("Enter your Jira username (email): ")
        
        if not token:
            import getpass
            token = getpass.getpass("Enter your Jira API token: ")
        
        # Initialize client
        jira_client = JiraClient(base_url, username, token)
        
        # Test connection
        if jira_client.test_connection():
            logger.info("Successfully connected to Jira.")
            return jira_client
        else:
            logger.error("Failed to connect to Jira. Please check your credentials and try again.")
            return None
    
    def run(self):
        """Run the interactive assistant."""
        print("\n" + "="*50)
        print("       📊 JIRA GEMINI ASSISTANT 🤖       ")
        print("="*50)
        print("Welcome to your Jira Gemini Assistant!")
        print("I can help you with Jira tickets, answer questions about your Jira environment,")
        print("search for issues, provide ticket details, and more.")
        print("\nType 'exit', 'quit', or 'bye' to end the session.")
        print("Type 'help' for usage tips.")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 YOU: ")
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nThank you for using Jira Gemini Assistant. Goodbye!")
                    break
                
                # Check for help command
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Check for clear command
                if user_input.lower() in ['clear', 'reset']:
                    self.assistant.conversation_history = []
                    print("\nConversation history cleared.")
                    continue
                
                # Process the query
                if user_input.strip():
                    print("\n🤖 ASSISTANT: Processing your query...", end="", flush=True)
                    print("\r" + " " * 50 + "\r", end="")  # Clear the processing message
                    
                    # Generate response
                    response = self.assistant.generate_response(user_input)
                    
                    # Print the response with formatting
                    self._print_formatted_response(response)
                    
                    # Check if a follow-up question is needed
                    follow_up = self.assistant.ask_follow_up_question(user_input, response)
                    if follow_up:
                        print(f"\n🤖 FOLLOW-UP: {follow_up}")
                        
                        # Get user's response to the follow-up
                        follow_up_input = input("\n🔍 YOU: ")
                        
                        if follow_up_input.strip() and follow_up_input.lower() not in ['exit', 'quit', 'bye']:
                            # Generate new response with the follow-up information
                            print("\n🤖 ASSISTANT: Processing with additional information...", end="", flush=True)
                            print("\r" + " " * 50 + "\r", end="")
                            
                            enhanced_query = f"{user_input}\n\nFollow-up clarification: {follow_up_input}"
                            response = self.assistant.generate_response(enhanced_query)
                            
                            # Print the response with formatting
                            self._print_formatted_response(response)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}")
                logger.error(f"Error in main loop: {str(e)}")
    
    def _print_formatted_response(self, response):
        """Print the assistant's response with formatting."""
        # Split response into lines
        lines = response.split('\n')
        
        # Process each line for formatting
        for i, line in enumerate(lines):
            # Format headings
            if re.match(r'^#+\s+.+', line):
                heading_level = len(re.match(r'^(#+)', line).group(1))
                if heading_level == 1:
                    print(f"\n\033[1;36m{line}\033[0m")  # Cyan, bold
                elif heading_level == 2:
                    print(f"\n\033[1;34m{line}\033[0m")  # Blue, bold
                else:
                    print(f"\n\033[1;33m{line}\033[0m")  # Yellow, bold
            
            # Format links
            elif re.search(r'https?://\S+', line):
                formatted_line = re.sub(r'(https?://\S+)', r'\033[4;36m\1\033[0m', line)  # Cyan, underlined
                print(formatted_line)
            
            # Format bullet points
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                print(f"\033[0;32m{line}\033[0m")  # Green
            
            # Format numbered lists
            elif re.match(r'^\d+\.\s+', line):
                print(f"\033[0;32m{line}\033[0m")  # Green
            
            # Format code blocks
            elif line.strip().startswith('```') or line.strip() == '```':
                if line.strip() == '```':
                    print("\033[0;37m```\033[0m")  # Gray
                else:
                    print(f"\033[0;37m{line}\033[0m")  # Gray
            
            # Format ticket references
            elif re.search(r'\b[A-Z]+-\d+\b', line):
                formatted_line = re.sub(r'(\b[A-Z]+-\d+\b)', r'\033[1;35m\1\033[0m', line)  # Magenta, bold
                print(formatted_line)
            
            # Everything else
            else:
                print(line)
    
    def _show_help(self):
        """Show help information."""
        help_text = """
        === JIRA GEMINI ASSISTANT HELP ===
        
        GENERAL COMMANDS:
        - exit, quit, bye: End the session
        - help: Show this help message
        - clear, reset: Clear conversation history
        
        QUERY EXAMPLES:
        1. Specific Ticket Queries:
           - "Tell me about PROJ-123"
           - "What's the status of PROJ-123?"
           - "Who is assigned to PROJ-123?"
           - "Show me the description of PROJ-123"
           - "List all comments on PROJ-123"
        
        2. Search Queries:
           - "Find all open bugs in project PROJ"
           - "Show me tickets assigned to John"
           - "List all high priority tickets created this week"
           - "Search for tickets containing 'login error'"
           - "Find all unassigned tickets in PROJ"
        
        3. Project Queries:
           - "Give me an overview of project PROJ"
           - "What are the active sprints in PROJ?"
           - "Who are the main contributors in PROJ?"
           - "How many open tickets are in PROJ?"
        
        4. Analysis Queries:
           - "What are the most common issues in PROJ?"
           - "Summarize the status of PROJ-123"
           - "Analyze the comments in PROJ-123"
           - "What's changed in PROJ-123 in the last week?"
        
        5. Specific Information:
           - "What is the workflow for PROJ?"
           - "Who is the project lead for PROJ?"
           - "When was PROJ-123 created?"
           - "What attachments are in PROJ-123?"
        
        Remember, you can ask follow-up questions to get more detailed information!
        """
        print(help_text)

def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Jira Gemini Assistant')
        parser.add_argument('--base-url', help='Jira base URL')
        parser.add_argument('--username', help='Jira username')
        parser.add_argument('--token', help='Jira API token')
        parser.add_argument('--project-id', help='Google Cloud project ID')
        parser.add_argument('--region', help='Google Cloud region')
        parser.add_argument('--model', help='Gemini model name')
        args = parser.parse_args()
        
        # Set environment variables if provided
        if args.base_url:
            os.environ['JIRA_BASE_URL'] = args.base_url
        if args.username:
            os.environ['JIRA_USERNAME'] = args.username
        if args.token:
            os.environ['JIRA_TOKEN'] = args.token
        if args.project_id:
            os.environ['PROJECT_ID'] = args.project_id
        if args.region:
            os.environ['REGION'] = args.region
        if args.model:
            os.environ['MODEL_NAME'] = args.model
        
        # Start the interactive assistant
        assistant = InteractiveJiraAssistant()
        assistant.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

















