#!/usr/bin/env python3
import logging
import os
import sys
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
# Ensure the base URL doesn't end with a trailing slash
CONFLUENCE_BASE_URL = os.environ.get("CONFLUENCE_BASE_URL", "https://your-company.atlassian.net").rstrip('/')
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
MAX_RESULTS_PER_QUERY = 100  # Fetch more content
DISABLE_SSL_VERIFICATION = True  # Always disable SSL verification

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
            base_url: The base URL of the Confluence instance (without trailing slash)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url
        self.rest_api_url = f"{self.base_url}/wiki/rest/api"
        self.auth = (username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "GeminiConfluenceChatbot/1.0 Python/Requests"
        }
        logger.info(f"Initialized Confluence client for {self.base_url}")
        logger.info(f"REST API URL: {self.rest_api_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.rest_api_url}/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=not DISABLE_SSL_VERIFICATION
            )
            
            logger.info(f"Connection test status code: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("Connection successful!")
                return True
            else:
                logger.error(f"Connection test failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
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
                f"{self.rest_api_url}/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=not DISABLE_SSL_VERIFICATION
            )
            response.raise_for_status()
            
            content = response.json()
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            return content
                
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve content by ID: {str(e)}")
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
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
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
    
    def search_content(self, cql=None, text_query=None, title=None, content_type="page", expand=None, limit=10, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            text_query: Simple text to search for (alternative to CQL)
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
            if not cql:
                query_parts = []
                if content_type:
                    query_parts.append(f"type={content_type}")
                    
                if title:
                    # Escape special characters in title
                    safe_title = title.replace('"', '\\"')
                    query_parts.append(f'title~"{safe_title}"')
                
                if text_query:
                    # Escape special characters in text
                    safe_text = text_query.replace('"', '\\"')
                    query_parts.append(f'text~"{safe_text}"')
                    
                if query_parts:
                    params["cql"] = " AND ".join(query_parts)
            else:
                params["cql"] = cql
                
            if expand:
                params["expand"] = expand
                
            logger.info(f"Searching for content with params: {params}")
            
            response = requests.get(
                f"{self.rest_api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=not DISABLE_SSL_VERIFICATION
            )
            
            # For debugging
            logger.info(f"Search status code: {response.status_code}")
            logger.info(f"Response content: {response.text[:200]}...")  # Log first 200 chars
            
            response.raise_for_status()
            
            results = response.json()
            logger.info(f"Search returned {len(results.get('results', []))} results")
            return results
                
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text[:500]}...")  # Log first 500 chars
            return {"results": []}
    
    def generic_search(self, query, limit=25):
        """
        Perform a generic search using multiple methods to ensure we get results.
        This is a more aggressive approach to finding content.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            list: List of search results
        """
        all_results = []
        result_ids = set()
        
        # Method 1: Search using the text query directly
        logger.info(f"Searching with direct text query: {query}")
        direct_results = self.search_content(
            text_query=query,
            content_type="page",
            limit=limit
        )
        
        for result in direct_results.get("results", []):
            result_id = result.get("id")
            if result_id and result_id not in result_ids:
                all_results.append(result)
                result_ids.add(result_id)
        
        logger.info(f"Direct text search returned {len(direct_results.get('results', []))} results")
        
        # Method 2: Search by title
        title_results = self.search_content(
            title=query,
            content_type="page",
            limit=limit
        )
        
        for result in title_results.get("results", []):
            result_id = result.get("id")
            if result_id and result_id not in result_ids:
                all_results.append(result)
                result_ids.add(result_id)
        
        logger.info(f"Title search returned {len(title_results.get('results', []))} results")
        
        # Method 3: If we still don't have enough results, try breaking the query into words
        if len(all_results) < 10:
            words = [w for w in re.findall(r'\b\w+\b', query) if len(w) > 3]
            for word in words[:5]:  # Use the first 5 significant words
                word_results = self.search_content(
                    text_query=word,
                    content_type="page",
                    limit=max(5, limit // len(words))
                )
                
                for result in word_results.get("results", []):
                    result_id = result.get("id")
                    if result_id and result_id not in result_ids:
                        all_results.append(result)
                        result_ids.add(result_id)
            
            logger.info(f"Word-based search returned additional results, total now: {len(all_results)}")
        
        logger.info(f"Generic search returned a total of {len(all_results)} results")
        return all_results
    
    def get_all_spaces(self):
        """Retrieve all spaces with pagination handling."""
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info("Retrieving all spaces")
        
        while True:
            try:
                response = requests.get(
                    f"{self.rest_api_url}/space",
                    auth=self.auth,
                    headers=self.headers,
                    params={"limit": limit, "start": start},
                    verify=not DISABLE_SSL_VERIFICATION
                )
                response.raise_for_status()
                
                spaces = response.json()
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
                    
            except requests.RequestException as e:
                logger.error(f"Error retrieving spaces: {str(e)}")
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
            max_output_tokens=2048,
        )
        
        # Create the prompt for analyzing the question
        prompt = f"""
        Analyze the following question and break it down into separate searchable queries if needed:
        
        USER QUESTION: "{question}"
        
        If this is a single question, return it as is.
        If it contains multiple separate questions, break it into distinct parts.
        
        FORMAT: Return a JSON array of strings, each representing a searchable query.
        Example: "How do I configure SAML and what are the security best practices?"
        Response: ["How do I configure SAML", "What are the SAML security best practices"]
        """
        
        # Get the response from Gemini
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            response_text = response.text.strip()
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
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            return [question]
    
    def extract_search_terms(self, question):
        """
        Extract key search terms from the question.
        
        Args:
            question: The question to extract terms from
            
        Returns:
            list: A list of search terms
        """
        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=1024,
        )
        
        prompt = f"""
        Extract 3-6 key search terms from this question that would help find relevant content in a knowledge base:
        
        QUESTION: "{question}"
        
        Rules:
        1. Include specific technical terms, product names, and unique identifiers
        2. Focus on nouns and technical concepts
        3. Include both technical abbreviations (like "SSO") and their full forms ("Single Sign-On")
        4. Preserve exact capitalization of technical terms
        
        Return a JSON array of strings.
        Example: "How do I configure JWT authentication for the API gateway?"
        Response: ["JWT authentication", "API gateway", "configure JWT", "JWT", "authentication configuration"]
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            response_text = response.text.strip()
            logger.info(f"Search terms extraction response: {response_text}")
            
            try:
                search_terms = json.loads(response_text)
                if isinstance(search_terms, list) and all(isinstance(term, str) for term in search_terms):
                    return search_terms
            except json.JSONDecodeError:
                pass
                
            # Fallback to simple extraction
            return self._simple_term_extraction(question)
        except Exception as e:
            logger.error(f"Error extracting search terms: {str(e)}")
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
    
    def search_relevant_content(self, confluence_client, question):
        """
        Search for content relevant to the question in Confluence using multiple strategies.
        
        Args:
            confluence_client: The Confluence client
            question: The question to search for
            
        Returns:
            list: A list of page content objects
        """
        logger.info(f"Searching for content relevant to: {question}")
        
        # First try a generic search with the whole question
        search_results = confluence_client.generic_search(question, limit=MAX_RESULTS_PER_QUERY)
        
        # If we didn't get enough results, try with extracted search terms
        if len(search_results) < 5:
            logger.info("Few results from generic search, trying with extracted terms")
            search_terms = self.extract_search_terms(question)
            logger.info(f"Extracted search terms: {search_terms}")
            
            for term in search_terms:
                term_results = confluence_client.generic_search(term, limit=20)
                search_results.extend([r for r in term_results if r.get("id") not in [res.get("id") for res in search_results]])
                
                if len(search_results) >= MAX_RESULTS_PER_QUERY:
                    break
        
        # Extract the content from each result
        page_contents = []
        for result in search_results:
            page_id = result.get("id")
            if page_id:
                page_content = confluence_client.get_page_content(page_id)
                if page_content:
                    page_contents.append(page_content)
        
        logger.info(f"Retrieved {len(page_contents)} pages of content")
        
        # If we still don't have any results, log an error
        if not page_contents:
            logger.error("No content found after search. Check Confluence access and search functionality.")
        
        return page_contents
    
    def generate_answer(self, question, contexts):
        """
        Generate an answer to the question based on the provided contexts.
        
        Args:
            question: The user's question
            contexts: A list of context objects with content and metadata
            
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
        
        # Check if we have any contexts
        if not contexts:
            logger.warning("No contexts provided for answer generation")
            return self._generate_no_content_response(question)
        
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
        
        # Create the prompt for generating the answer
        prompt = f"""
        You are a helpful assistant with access to Confluence documentation. Your task is to answer questions based on the provided documentation content.

        IMPORTANT INSTRUCTIONS:
        1. Use ONLY the information provided in the documents below to answer the question
        2. If the provided documents contain the information, provide a detailed, helpful answer
        3. Format your answer nicely with markdown headings, bullet points, or numbered lists as appropriate
        4. Include relevant code examples, configurations, or technical details from the documentation if available
        5. If the documents don't contain enough information, acknowledge what you DO know, then briefly suggest what additional information might be helpful
        6. NEVER invent information that isn't in the documents
        
        CONFLUENCE DOCUMENTATION:
        {context_text}
        
        USER QUESTION:
        {question}
        
        Answer the question based on the documentation provided above, and include a "References" section at the end listing the documents you used.
        """
        
        # Get the response from Gemini
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            answer = response.text
            
            # If references weren't included, append them
            if "## References" not in answer and "# References" not in answer:
                answer += "\n\n## References\n"
                for i, ref in enumerate(references, 1):
                    answer += f"{i}. [{ref['title']}]({ref['url']})\n"
            
            logger.info(f"Generated answer of length: {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating your answer. Please try rephrasing your question or contact support if the issue persists. Error details: {str(e)}"
    
    def _generate_no_content_response(self, question):
        """Generate a response when no content is found."""
        logger.info("Generating no-content response")
        
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=2048,
        )
        
        prompt = f"""
        You are a helpful assistant that works with a Confluence knowledge base. 
        
        Unfortunately, no relevant documentation was found for the following question:
        
        "{question}"
        
        Create a helpful response that:
        1. Acknowledges that you don't have specific documentation on this topic
        2. Suggests possible reasons (like the information might be in a different system, or using different terminology)
        3. Asks for clarification or additional details that might help find relevant documentation
        4. Maintains a helpful, professional tone
        
        Do NOT suggest specific technical solutions since you don't have access to the relevant documentation.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating no-content response: {str(e)}")
            return "I couldn't find any relevant information in the knowledge base. Could you try rephrasing your question with different terminology or provide more context about what you're looking for?"

class GeminiConfluenceChatbot:
    """Main chatbot class integrating Confluence and Gemini."""
    
    def __init__(self, initialize=True):
        """Initialize the chatbot."""
        if initialize:
            self.confluence_client = ConfluenceClient(
                base_url=CONFLUENCE_BASE_URL,
                username=CONFLUENCE_USERNAME,
                api_token=CONFLUENCE_API_TOKEN
            )
            self.gemini_manager = GeminiManager()
            self.conversation_history = []
            
            # Test connection
            if not self.confluence_client.test_connection():
                logger.error("Failed to connect to Confluence. Check credentials and connection.")
                print("WARNING: Failed to connect to Confluence. Please check your credentials and connection.")
            else:
                logger.info("Connected to Confluence successfully")
                
            logger.info("GeminiConfluenceChatbot initialized successfully")
        else:
            logger.info("GeminiConfluenceChatbot created without initialization")
    
    def process_question(self, question):
        """
        Process a user question and generate a response.
        
        Args:
            question: The user's question
            
        Returns:
            str: The chatbot's response
        """
        logger.info(f"Processing question: {question}")
        
        start_time = time.time()
        
        # Check if this is a follow-up question
        is_followup = len(self.conversation_history) > 0
        
        # For follow-up questions, include previous context in the search
        if is_followup:
            logger.info("Including previous context in search")
            # Get the most recent conversation
            prev_question, _ = self.conversation_history[-1]
            
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
        
        for subq in subquestions:
            logger.info(f"Processing subquestion: {subq}")
            
            # Search for relevant content
            contexts = self.gemini_manager.search_relevant_content(
                self.confluence_client, 
                subq
            )
            
            # Add contexts to the collected list, avoiding duplicates
            existing_ids = {ctx.get("metadata", {}).get("id") for ctx in all_contexts}
            for ctx in contexts:
                ctx_id = ctx.get("metadata", {}).get("id")
                if ctx_id not in existing_ids:
                    all_contexts.append(ctx)
                    existing_ids.add(ctx_id)
        
        logger.info(f"Collected {len(all_contexts)} unique contexts across all subquestions")
        
        # Generate an answer from the contexts
        answer = self.gemini_manager.generate_answer(question, all_contexts)
        
        # Add to conversation history
        self.conversation_history.append((question, answer))
        
        # Limit history to recent conversations
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
            
        end_time = time.time()
        logger.info(f"Question processed in {end_time - start_time:.2f} seconds")
        
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
    
    # Simple HTML template for the chat interface
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gemini Confluence Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chatbox { height: 500px; border: 1px solid #ccc; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
            #user-input { width: 85%; padding: 8px; }
            #send-button { width: 10%; padding: 8px; }
            .user-message { background-color: #e1f5fe; padding: 8px; margin: 5px 0; border-radius: 5px; }
            .bot-message { background-color: #f1f1f1; padding: 8px; margin: 5px 0; border-radius: 5px; }
            .thinking { color: #888; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>Gemini Confluence Chatbot</h1>
        <div id="chatbox"></div>
        <div>
            <input type="text" id="user-input" placeholder="Ask a question...">
            <button id="send-button">Send</button>
        </div>
        
        <script>
            const chatbox = document.getElementById('chatbox');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            
            function addMessage(message, isUser) {
                const msgDiv = document.createElement('div');
                msgDiv.className = isUser ? 'user-message' : 'bot-message';
                msgDiv.innerHTML = message;
                chatbox.appendChild(msgDiv);
                chatbox.scrollTop = chatbox.scrollHeight;
            }
            
            function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                userInput.value = '';
                
                // Add thinking indicator
                const thinkingDiv = document.createElement('div');
                thinkingDiv.className = 'thinking bot-message';
                thinkingDiv.textContent = 'Thinking...';
                chatbox.appendChild(thinkingDiv);
                chatbox.scrollTop = chatbox.scrollHeight;
                
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: message })
                })
                .then(response => response.json())
                .then(data => {
                    // Remove thinking indicator
                    chatbox.removeChild(thinkingDiv);
                    
                    if (data.status === 'success') {
                        // Convert markdown to HTML (simple version)
                        let html = data.answer
                            .replace(/\\n/g, '<br>')
                            .replace(/#{3,6}\s*(.*?)\s*$/gm, '<h3>$1</h3>')
                            .replace(/#{2}\s*(.*?)\s*$/gm, '<h2>$1</h2>')
                            .replace(/#{1}\s*(.*?)\s*$/gm, '<h1>$1</h1>')
                            .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
                            .replace(/\*(.*?)\*/g, '<i>$1</i>')
                            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
                            .replace(/`([^`]+)`/g, '<code>$1</code>')
                            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>')
                            .replace(/^\s*[\*\-]\s+(.*?)$/gm, '<li>$1</li>')
                            .replace(/^\s*(\d+)\.\s+(.*?)$/gm, '<li>$1. $2</li>');
                            
                        addMessage(html, false);
                    } else {
                        addMessage('Error: ' + data.message, false);
                    }
                })
                .catch(error => {
                    // Remove thinking indicator
                    chatbox.removeChild(thinkingDiv);
                    addMessage('Error connecting to the server. Please try again.', false);
                    console.error('Error:', error);
                });
            }
            
            sendButton.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def index():
        return render_template_string(html_template)
    
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
        missing_vars = []
        if not CONFLUENCE_BASE_URL or CONFLUENCE_BASE_URL == "https://your-company.atlassian.net":
            missing_vars.append("CONFLUENCE_BASE_URL")
        if not CONFLUENCE_USERNAME:
            missing_vars.append("CONFLUENCE_USERNAME")
        if not CONFLUENCE_API_TOKEN:
            missing_vars.append("CONFLUENCE_API_TOKEN")
            
        if missing_vars:
            print("WARNING: The following environment variables are not set:")
            for var in missing_vars:
                print(f"- {var}")
            print("\nYou can set them using the following commands:")
            print('export CONFLUENCE_BASE_URL="https://your-company.atlassian.net"')
            print('export CONFLUENCE_USERNAME="your-email@company.com"')
            print('export CONFLUENCE_API_TOKEN="your-api-token"')
            
            # Prompt user if they want to continue
            if input("\nDo you want to continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
                
        # Check for API mode flag
        api_mode = len(sys.argv) > 1 and sys.argv[1] == "--api"
        
        # Create the chatbot
        chatbot = GeminiConfluenceChatbot()
        
        if api_mode:
            # Start the Flask API
            try:
                from flask import Flask
                app = create_flask_app(chatbot)
                port = int(os.environ.get("PORT", 5000))
                print(f"Starting API server on port {port}...")
                print(f"Access the chat interface at http://localhost:{port}/")
                app.run(host="0.0.0.0", port=port)
            except ImportError:
                print("Flask is not installed. Install it with: pip install flask")
                sys.exit(1)
        else:
            # Start the interactive console chat
            chatbot.chat()
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        sys.exit(1)






























#!/usr/bin/env python3
"""
Basic Remedy Chatbot - A simple, direct implementation for querying BMC Remedy incidents.
"""

import requests
import logging
import os
import sys
import urllib3
import re
import json
from datetime import datetime, timedelta
import getpass

# Disable SSL warnings
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

# Remedy server URL - CHANGE THIS TO YOUR SERVER
REMEDY_SERVER = "https://cmegroup-restapi.onbmc.com"

class RemedyClient:
    """Simple client for BMC Remedy REST API operations."""
    
    def __init__(self, server_url=REMEDY_SERVER, username=None, password=None):
        """Initialize the Remedy client."""
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.token_type = "AR-JWT"
        
        # Always disable SSL verification for enterprise environments
        self.ssl_verify = False
        
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self):
        """Log in to Remedy and get authentication token."""
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
                return True
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                print(f"Login failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            print(f"Login error: {str(e)}")
            return False
    
    def get_incident(self, incident_id):
        """Get a specific incident by its ID."""
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Fetching incident: {incident_id}")
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
    
    def get_incidents_by_date(self, date_str):
        """Get all incidents submitted on a specific date."""
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        # Parse the date string
        try:
            if date_str.lower() == 'today':
                date_obj = datetime.now()
            elif date_str.lower() == 'yesterday':
                date_obj = datetime.now() - timedelta(days=1)
            else:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
            end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
            
            logger.info(f"Fetching incidents for date: {date_obj.strftime('%Y-%m-%d')}")
            
            # Create qualified query
            qualified_query = f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""
            
            # Fields to retrieve
            fields = [
                "Assignee", "Incident Number", "Description", "Status", "Owner",
                "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
                "Priority", "Environment", "Summary", "Support Group Name"
            ]
            
            # Get the incidents
            result = self.query_form("HPD:Help Desk", qualified_query, fields)
            if result and "entries" in result:
                logger.info(f"Retrieved {len(result['entries'])} incidents for date {date_str}")
                return result["entries"]
            else:
                logger.warning(f"No incidents found for date {date_str} or error occurred")
                return []
        except Exception as e:
            logger.error(f"Error processing date {date_str}: {str(e)}")
            return []
    
    def get_incidents_by_status(self, status):
        """Get incidents by their status."""
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
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []
    
    def get_incidents_by_assignee(self, assignee):
        """Get incidents assigned to a specific person."""
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
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents assigned to {assignee}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found assigned to {assignee} or error occurred")
            return []
    
    def get_incidents_by_owner_group(self, owner_group):
        """Get incidents owned by a specific group."""
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
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents owned by group {owner_group}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found owned by group {owner_group} or error occurred")
            return []
    
    def query_form(self, form_name, qualified_query=None, fields=None, limit=100):
        """Query a Remedy form with optional filters and field selection."""
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


class RemedyChatbot:
    """Simple chatbot for interacting with Remedy."""
    
    def __init__(self, remedy_username=None, remedy_password=None):
        """Initialize the chatbot."""
        self.remedy = RemedyClient(username=remedy_username, password=remedy_password)
    
    def login(self):
        """Login to Remedy."""
        return self.remedy.login()
    
    def format_incident(self, incident):
        """Format a single incident for display."""
        if not incident or "values" not in incident:
            return "Incident data not available."
        
        values = incident["values"]
        lines = [
            f"Incident Number: {values.get('Incident Number', 'N/A')}",
            f"Summary: {values.get('Summary', 'N/A')}",
            f"Status: {values.get('Status', 'N/A')}",
            f"Priority: {values.get('Priority', 'N/A')}",
            f"Assigned To: {values.get('Assignee', 'N/A')}",
            f"Owner Group: {values.get('Owner Group', 'N/A')}",
            f"Submitter: {values.get('Submitter', 'N/A')}",
            f"Submit Date: {values.get('Submit Date', 'N/A')}",
            f"Impact: {values.get('Impact', 'N/A')}"
        ]
        
        if values.get('Description'):
            lines.append(f"\nDescription: {values.get('Description')}")
        
        return "\n".join(lines)
    
    def format_incidents_summary(self, incidents):
        """Format a summary of multiple incidents."""
        if not incidents:
            return "No incidents found."
        
        lines = [f"Found {len(incidents)} incidents:"]
        
        for i, incident in enumerate(incidents, 1):
            if "values" not in incident:
                continue
            
            values = incident["values"]
            lines.append(f"\n{i}. {values.get('Incident Number', 'N/A')} - {values.get('Summary', 'N/A')}")
            lines.append(f"   Status: {values.get('Status', 'N/A')} | Priority: {values.get('Priority', 'N/A')}")
            lines.append(f"   Assignee: {values.get('Assignee', 'N/A')}")
            lines.append(f"   Submit Date: {values.get('Submit Date', 'N/A')}")
        
        return "\n".join(lines)
    
    def process_query(self, query):
        """Process a user query and generate a response."""
        # Check for specific incident number
        inc_match = re.search(r'(INC\d{9,}|\d{9,})', query, re.IGNORECASE)
        if inc_match:
            incident_id = inc_match.group(1)
            if not incident_id.upper().startswith("INC"):
                incident_id = "INC" + incident_id
            
            incident = self.remedy.get_incident(incident_id)
            if incident:
                return f"Here are the details for {incident_id}:\n\n{self.format_incident(incident)}"
            else:
                return f"I couldn't find incident {incident_id} in the system."
        
        # Check for date queries
        date_match = re.search(r'(today|yesterday|\d{4}-\d{2}-\d{2})', query, re.IGNORECASE)
        if date_match or "happened" in query.lower():
            date_str = date_match.group(1) if date_match else "today"
            incidents = self.remedy.get_incidents_by_date(date_str)
            return f"Here's a summary of incidents from {date_str}:\n\n{self.format_incidents_summary(incidents)}"
        
        # Check for status queries
        status_keywords = {
            "open": "Open",
            "closed": "Closed",
            "resolved": "Resolved",
            "in progress": "In Progress",
            "pending": "Pending"
        }
        
        for keyword, status in status_keywords.items():
            if keyword in query.lower():
                incidents = self.remedy.get_incidents_by_status(status)
                return f"Here's a summary of {status} incidents:\n\n{self.format_incidents_summary(incidents)}"
        
        # Check for assignee queries
        assignee_match = re.search(r'assigned to (.+?)(?:$|\?|\.)', query, re.IGNORECASE)
        if assignee_match:
            assignee = assignee_match.group(1).strip()
            incidents = self.remedy.get_incidents_by_assignee(assignee)
            return f"Here's a summary of incidents assigned to {assignee}:\n\n{self.format_incidents_summary(incidents)}"
        
        # Check for owner group queries
        group_match = re.search(r'(?:group|team) (.+?)(?:$|\?|\.)', query, re.IGNORECASE)
        if group_match:
            group = group_match.group(1).strip()
            incidents = self.remedy.get_incidents_by_owner_group(group)
            return f"Here's a summary of incidents owned by group {group}:\n\n{self.format_incidents_summary(incidents)}"
        
        # Default response for unrecognized queries
        return "I'm not sure how to answer that. You can ask about specific incidents by ID, incidents from a particular date, or incidents with a specific status, assignee, or owner group."
    
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
        
        print("Login successful! You can now ask questions.\n")
        print("Example queries:")
        print(" - 'Show me incident INC000012345'")
        print(" - 'What incidents happened yesterday?'")
        print(" - 'Show me all open incidents'")
        print(" - 'What incidents are assigned to John Smith?'")
        print(" - 'Show me incidents for the TOCC Support group'")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Remedy Chatbot. Goodbye!")
                break
            
            print("\nProcessing...")
            response = self.process_query(user_input)
            print(f"\nChatbot: {response}")


if __name__ == "__main__":
    # Get credentials from environment or user input
    remedy_username = os.environ.get("REMEDY_USERNAME")
    remedy_password = os.environ.get("REMEDY_PASSWORD")
    
    chatbot = RemedyChatbot(remedy_username, remedy_password)
    chatbot.run_chat_loop()
