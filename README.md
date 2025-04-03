claude - 

#!/usr/bin/env python3
"""
COPPER View to API Mapper
--------------------------
This tool searches Confluence for COPPER-related documentation,
processes it with Gemini AI, and provides answers about mapping
database views to REST APIs.
"""

import logging
import os
import sys
import json
import re
from datetime import datetime

# Confluence imports
import requests
from html.parser import HTMLParser
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
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://your-confluence-instance.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")

class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content"""
    def __init__(self):
        super().__init__()
        self.text = ""
    
    def handle_data(self, data):
        self.text += data + " "


class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
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
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.api_url}/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=True
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info("Connection to Confluence successful!")
                return True
            else:
                logger.warning(f"Empty response received during connection test")
                return False  # Still consider it a success if status code is OK
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def search_content(self, cql, limit=25, expand=None):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            limit: Maximum number of results to return
            expand: Comma separated list of properties to expand
        """
        try:
            query_params = {
                "cql": cql,
                "limit": limit
            }
            
            if expand:
                query_params["expand"] = expand
            
            logger.info(f"Searching for content with params: {query_params}")
            response = requests.get(
                f"{self.api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params=query_params,
                verify=True
            )
            response.raise_for_status()
            
            # Handle empty response
            if not response.text.strip():
                logger.warning("Empty response received when searching content")
                return {"results": []}
                
            try:
                search_results = response.json()
                logger.info(f"Search returned {len(search_results.get('results', []))} results")
                return search_results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for search: {str(e)}")
                logger.error(f"Response content: {response.text}")
                return {"results": []}
                
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            return {"results": []}
    
    def get_page_content(self, page_id, expand=None):
        """
        Get the content of a page in a suitable format for NLP.
        This extracts and processes the content to be more suitable for embeddings.
        
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
                "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process with HTML filter
            html_filter = HTMLTextExtractor()
            html_filter.feed(content)
            plain_text = html_filter.text
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
            
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
            params = {}
            if expand:
                params["expand"] = expand
                
            logger.info(f"Fetching content with ID: {content_id}")
            response = requests.get(
                f"{self.api_url}/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False  # In a production environment, this should be True
            )
            response.raise_for_status()
            
            # Handle empty response
            if not response.text.strip():
                logger.warning(f"Empty response received when retrieving content ID: {content_id}")
                return None
                
            try:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
                return content
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for content ID {content_id}: {str(e)}")
                logger.error(f"Response content: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Connection succeeded but failed to get content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def find_copper_pages(self, limit=25):
        """Find all pages related to COPPER database/API."""
        search_query = 'text ~ "COPPER database" OR text ~ "COPPER API" OR text ~ "COPPER views" OR title ~ "COPPER"'
        return self.search_content(search_query, limit=limit, expand="body.storage")


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
            You are the COPPER Assistant, an expert on mapping database views to REST APIs.
            You help answer questions about the COPPER database system, its views, and corresponding API endpoints.
            Focus on providing accurate, helpful answers about view-to-API mappings, utilizing the context provided.
            If the context doesn't contain the information needed, admit what you don't know rather than making up facts.
            """
            
            # Craft the full prompt with context
            full_prompt = system_prompt + "\n\n"
            
            if copper_context:
                full_prompt += "COPPER CONTEXT:\n" + copper_context + "\n\n"
                
            full_prompt += f"USER QUESTION: {prompt}\n\nResponse:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.2,  # Lower temperature for more accurate/consistent responses
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
                return "I'm sorry, I couldn't generate a response for your question. Please try rephrasing or providing more details."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"


class CopperAssistant:
    """Main class that coordinates between Confluence and Gemini."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token):
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.gemini = GeminiAssistant()
        self.copper_pages = []
        self.copper_context = ""
        logger.info("Initialized Copper Assistant")
        
    def initialize(self):
        """Initialize by testing connections and gathering initial COPPER context."""
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and URL.")
            return False
            
        logger.info("Loading COPPER knowledge base...")
        self.load_copper_knowledge()
        return True
        
    def load_copper_knowledge(self):
        """Load and cache COPPER-related knowledge from Confluence."""
        # Find pages related to COPPER
        search_results = self.confluence.find_copper_pages(limit=25)
        
        if not search_results or "results" not in search_results or not search_results["results"]:
            logger.warning("No COPPER-related pages found in Confluence")
            return
            
        # Store basic page info for later querying
        self.copper_pages = []
        
        # Limit to top 10 most relevant pages to avoid overwhelming context
        for page in search_results["results"][:10]:
            logger.info(f"Processing COPPER page: {page.get('title', 'Unknown')}")
            page_content = self.confluence.get_page_content(page["id"])
            
            if page_content:
                self.copper_pages.append({
                    "id": page["id"],
                    "title": page["title"],
                    "content": page_content["content"],
                    "url": page_content["metadata"]["url"]
                })
                
        logger.info(f"Loaded {len(self.copper_pages)} COPPER-related pages")
        
    def extract_relevant_content(self, query):
        """Extract content from cached pages that's most relevant to the query."""
        if not self.copper_pages:
            return "No COPPER documentation found in Confluence."
            
        relevant_content = []
        
        # Create a simple keyword match scoring system
        keywords = set(re.findall(r'\b\w+\b', query.lower()))
        if len(keywords) < 2:  # If query is too short, add COPPER-related terms
            keywords.update(["copper", "api", "database", "view", "endpoint", "rest", "mapping"])
            
        for page in self.copper_pages:
            # Count keyword occurrences for simple relevance scoring
            score = sum(1 for keyword in keywords if keyword in page["content"].lower())
            if score > 0:
                # Extract a section of text around the highest concentration of keywords
                # Simplified approach: just grab paragraphs containing keywords
                paragraphs = page["content"].split("\n\n")
                relevant_paragraphs = []
                
                for para in paragraphs:
                    para_score = sum(1 for keyword in keywords if keyword in para.lower())
                    if para_score > 0:
                        relevant_paragraphs.append(para)
                        
                        # Limit to top 3 relevant paragraphs per page
                        if len(relevant_paragraphs) >= 3:
                            break
                            
                if relevant_paragraphs:
                    relevant_content.append(
                        f"--- FROM PAGE: {page['title']} ---\n"
                        + "\n".join(relevant_paragraphs[:3])
                        + f"\n(Source: {page['url']})\n"
                    )
        
        if not relevant_content:
            # If no specific content matched, use titles and summaries
            relevant_content = [
                f"COPPER Pages in Confluence:\n" +
                "\n".join([f"- {page['title']}" for page in self.copper_pages])
            ]
            
        return "\n\n".join(relevant_content)
    
    def answer_question(self, question):
        """Answer a question about COPPER views, APIs, or mappings."""
        logger.info(f"Processing question: {question}")
        
        # Extract relevant content based on the question
        copper_context = self.extract_relevant_content(question)
        
        # Generate response using Gemini
        response = self.gemini.generate_response(question, copper_context)
        
        return response


def main():
    """Main entry point for the COPPER Assistant."""
    logger.info("Starting COPPER Assistant")
    
    # Check for required environment variables
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN:
        logger.error("Missing Confluence credentials. Please set CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN environment variables.")
        print("Error: Missing Confluence credentials. Please set the required environment variables.")
        return
        
    # Initialize the assistant
    assistant = CopperAssistant(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
    
    if not assistant.initialize():
        logger.error("Failed to initialize COPPER Assistant.")
        print("Error: Failed to initialize. Please check the logs for details.")
        return
        
    print("\n===== COPPER Database-to-API Mapping Assistant =====")
    print("Ask questions about COPPER database views and their API mappings.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("Question: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            answer = assistant.answer_question(user_input)
            print("\nAnswer:")
            print("-------")
            print(answer)
            print("-------\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

















gemini - Okay, let's build this step-by-step. We'll combine the Confluence data fetching logic and the Gemini interaction logic into a single Python script. The core idea is:

Configure: Set up credentials and parameters for both Confluence and Vertex AI/Gemini using environment variables.

Search Confluence: Take a user question, formulate a Confluence Query Language (CQL) search that includes terms related to "Copper" along with the user's query terms, and retrieve the most relevant page IDs.

Fetch Content: Get the actual text content of those relevant Confluence pages.

Build Prompt: Construct a prompt for Gemini that includes:

System instructions (telling it to act as a helpful assistant using the provided context).

The fetched Confluence content as context.

The original user question.

Query Gemini: Send the prompt to Gemini via the Vertex AI SDK.

Return Response: Output Gemini's answer.

Here is the combined Python script.

Prerequisites:

Install Libraries:

pip install google-cloud-aiplatform google-auth requests beautifulsoup4 html5lib # Added beautifulsoup4 and html5lib for better HTML parsing
Use code with caution.
Bash
Authentication:

Google Cloud (Gemini): Make sure you've run gcloud auth application-default login in your environment, or set the GOOGLE_APPLICATION_CREDENTIALS environment variable.

Confluence: Set the following environment variables:

CONFLUENCE_URL: Your Confluence instance URL (e.g., https://yourcompany.atlassian.net)

CONFLUENCE_USERNAME: Your Confluence email/username.

CONFLUENCE_API_TOKEN: Your Confluence API token.

Google Cloud Project: Set the following environment variables:

PROJECT_ID: Your Google Cloud Project ID.

REGION: The region for Vertex AI (e.g., us-central1).

MODEL_NAME: The Gemini model name (e.g., gemini-1.5-flash-001).

#!/usr/bin/env python3

import logging
import os
import sys
import json
import requests
import argparse
from requests.auth import HTTPBasicAuth
import urllib3
from bs4 import BeautifulSoup # Using BeautifulSoup for more robust HTML parsing
from datetime import datetime

# --- Google Cloud / Vertex AI / Gemini Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, SafetySetting, HarmCategory

# --- Disable SSL Warnings (Use with caution, typical in some corporate envs) ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("copper_rag_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CopperConfluenceGeminiRAG")

# --- Configuration (Environment Variables) ---
# Confluence Config
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME")
CONfluence_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN")
# Vertex AI Config
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash-001") # Use a capable model

# --- Input Validation ---
if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONfluence_API_TOKEN, PROJECT_ID, REGION, MODEL_NAME]):
    logger.error("Missing required environment variables. Please set CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, PROJECT_ID, REGION, MODEL_NAME.")
    sys.exit(1)

# =========================================
# Confluence Client Class
# =========================================
class ConfluenceClient:
    """Client for Confluence REST API operations focused on RAG."""

    def __init__(self, base_url, username, api_token):
        """Initialize the Confluence client."""
        if not base_url or not username or not api_token:
             raise ValueError("Confluence URL, username, and API token are required.")
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CopperConfluenceGeminiRAG/1.0 Python/Requests"
        }
        logger.info(f"Initialized Confluence client for {self.base_url}")
        self._test_connection() # Test connection on init

    def _test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.api_url}/space", # Use /space endpoint which requires less data
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False, # Insecure, consider using certifi or internal CA bundle
                timeout=10 # Add a timeout
            )
            response.raise_for_status()
            logger.info("Confluence connection successful!")
            return True
        except requests.RequestException as e:
            logger.error(f"Confluence connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text[:500]}") # Log part of the response
            # Re-raise a more specific error or exit if critical
            raise ConnectionError(f"Failed to connect to Confluence: {str(e)}") from e


    def search_content(self, user_query, copper_filter_terms=["copper"], content_type="page", limit=5, space_key=None):
        """Search for content using CQL, adding Copper-specific filters."""
        try:
            # Build CQL Query
            # Basic user query terms + copper filter + content type
            # Simple approach: treat user query as keywords
            query_parts = [f'text ~ "{term}"' for term in user_query.split()]
            filter_parts = [f'text ~ "{term}"' for term in copper_filter_terms]

            cql_parts = query_parts + filter_parts
            if content_type:
                cql_parts.append(f'type="{content_type}"')
            if space_key:
                 cql_parts.append(f'space="{space_key}"') # Allow searching specific space

            cql_query = " AND ".join(cql_parts)

            params = {
                "cql": cql_query,
                "limit": limit,
                "expand": "body.storage,version,space" # Expand to get content directly if needed, but get_page_content is often better
            }
            logger.info(f"Searching Confluence with CQL: {cql_query} (limit: {limit})")

            response = requests.get(
                f"{self.api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False, # Consider security implications
                timeout=30 # Longer timeout for search
            )
            response.raise_for_status()

            # Handle empty or invalid JSON response
            try:
                results = response.json()
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response from search. Status: {response.status_code}. Content: {response.text[:500]}")
                 return {"results": []} # Return empty results

            logger.info(f"Confluence search returned {len(results.get('results', []))} results")
            return results

        except requests.RequestException as e:
            logger.error(f"Failed to search Confluence content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text[:500]}")
            return {"results": []} # Return empty list on error
        except Exception as e:
             logger.error(f"An unexpected error occurred during Confluence search: {str(e)}", exc_info=True)
             return {"results": []}


    def get_page_content(self, page_id):
        """Get and clean the text content of a specific page."""
        try:
            logger.info(f"Fetching content for page ID: {page_id}")
            response = requests.get(
                f"{self.api_url}/content/{page_id}",
                auth=self.auth,
                headers=self.headers,
                params={"expand": "body.storage,space"}, # Need body.storage for content
                verify=False,
                timeout=15
            )
            response.raise_for_status()

            try:
                page_data = response.json()
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response for page {page_id}. Status: {response.status_code}. Content: {response.text[:500]}")
                 return None # Indicate failure

            raw_html = page_data.get("body", {}).get("storage", {}).get("value", "")
            if not raw_html:
                logger.warning(f"No content found in body.storage for page ID: {page_id}")
                return None

            # Use BeautifulSoup to parse and extract text
            soup = BeautifulSoup(raw_html, 'html5lib') # Use html5lib for robustness
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Get text, stripping leading/trailing whitespace from lines
            lines = (line.strip() for line in soup.get_text().splitlines())
            # Join lines, keeping paragraph breaks (double newline -> single newline)
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            plain_text = '\n'.join(chunk for chunk in chunks if chunk)

            page_title = page_data.get("title", "Unknown Title")
            space_key = page_data.get("space", {}).get("key", "UNK")
            page_url = f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}"

            logger.info(f"Successfully extracted text content for page: '{page_title}' (ID: {page_id})")

            # Return structured content
            return {
                "id": page_id,
                "title": page_title,
                "url": page_url,
                "text": plain_text
            }

        except requests.RequestException as e:
            logger.error(f"Failed to get content for page ID {page_id}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text[:500]}")
            return None # Indicate failure
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching page {page_id}: {str(e)}", exc_info=True)
            return None

# =========================================
# Gemini Interaction Class
# =========================================
class GeminiRAG:
    """Handles interaction with Gemini model for RAG."""

    def __init__(self, project_id, location, model_name):
        try:
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name} in {project_id}/{location}")

            # Configure safety settings (adjust as needed for your content)
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            # Default Generation Config (can be overridden)
            self.default_generation_config = GenerationConfig(
                temperature=0.3, # Lower temp for factual RAG
                top_p=0.9,
                max_output_tokens=4096 # Adjust based on model and needs
            )

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI or Gemini Model: {e}", exc_info=True)
            raise

    def generate_answer(self, question, context, generation_config=None, stream=True):
        """Generates an answer using Gemini based on the question and context."""
        if not question:
            logger.error("Question cannot be empty.")
            return "Error: Question is missing."
        if not context:
            logger.warning("Context is empty. Asking Gemini without context.")
            context_str = "No context provided."
            system_instruction_text = "You are a helpful assistant. Please answer the user's question."
        else:
             # Format context with source information for clarity
            context_str = "\n\n---\n\n".join([
                f"Source Page (ID: {ctx['id']}, Title: {ctx['title']}):\n{ctx['text']}"
                for ctx in context
            ])
            system_instruction_text = (
                "You are a helpful assistant knowledgeable about Copper APIs based on internal Confluence documentation. "
                "Answer the user's question using *only* the provided context from Confluence pages below. "
                "If the context doesn't contain the answer, clearly state that the information wasn't found in the provided documents. "
                "Be concise and accurate. Do not make up information."
            )


        # Limit context size (example: rough character limit, needs refinement based on tokens)
        MAX_CONTEXT_CHARS = 25000 # Adjust based on model's token limit and typical content size
        if len(context_str) > MAX_CONTEXT_CHARS:
             logger.warning(f"Context length ({len(context_str)} chars) exceeds limit ({MAX_CONTEXT_CHARS}). Truncating.")
             context_str = context_str[:MAX_CONTEXT_CHARS] + "\n... [Context Truncated]"


        full_prompt = [
            Part.from_text(context_str),
            Part.from_text(f"\n\nUser Question: {question}\n\nAnswer:")
        ]

        gen_config_to_use = generation_config if generation_config else self.default_generation_config

        logger.info(f"Sending prompt to Gemini (System Instruction + Context Chars: {len(context_str)} + Question)")
        # logger.debug(f"Full prompt context: {context_str[:500]}...") # Log beginning of context if needed

        try:
            if stream:
                logger.info("Generating response (streaming)...")
                response_stream = self.model.generate_content(
                    full_prompt,
                    generation_config=gen_config_to_use,
                    safety_settings=self.safety_settings,
                    stream=True,
                    system_instruction=system_instruction_text # Use system instruction parameter
                )
                full_response_text = ""
                print("Gemini Response: ", end="")
                for chunk in response_stream:
                     if chunk.candidates and chunk.candidates[0].content.parts:
                         part_text = chunk.candidates[0].content.parts[0].text
                         print(part_text, end="", flush=True)
                         full_response_text += part_text
                print() # New line after stream
                logger.info(f"Finished streaming response. Length: {len(full_response_text)}")
                return full_response_text
            else:
                logger.info("Generating response (non-streaming)...")
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=gen_config_to_use,
                    safety_settings=self.safety_settings,
                    stream=False,
                     system_instruction=system_instruction_text
                )
                if response.candidates and response.candidates[0].content.parts:
                     full_response_text = response.candidates[0].content.parts[0].text
                     logger.info(f"Received non-streaming response. Length: {len(full_response_text)}")
                     return full_response_text
                else:
                     # Handle cases where response might be blocked or empty
                     logger.warning(f"Gemini response was empty or blocked. Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
                     # Check for safety ratings if available
                     safety_info = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'
                     logger.warning(f"Safety Feedback: {safety_info}")
                     return "Error: Gemini did not return a valid response. It might have been blocked due to safety settings or other reasons."

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}", exc_info=True)
            # Check for specific API call errors if using google.api_core exceptions
            # from google.api_core.exceptions import GoogleAPICallError
            # if isinstance(e, GoogleAPICallError):
            #     logger.error(f"API Error Details: {e}")
            return f"Error: Failed to get response from Gemini - {str(e)}"

# =========================================
# Main Execution Logic
# =========================================
def main():
    parser = argparse.ArgumentParser(description="Ask questions about Copper using Confluence and Gemini.")
    parser.add_argument("question", type=str, help="The question you want to ask about Copper.")
    parser.add_argument("--space", type=str, default=None, help="Optional: Confluence Space Key to limit search.")
    parser.add_argument("--limit", type=int, default=3, help="Number of Confluence pages to use as context.")
    parser.add_argument("--nostream", action="store_true", help="Use non-streaming response from Gemini.")
    args = parser.parse_args()

    logger.info("Starting RAG process...")
    logger.info(f"User Question: {args.question}")

    try:
        # 1. Initialize Clients
        confluence = ConfluenceClient(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONfluence_API_TOKEN)
        gemini = GeminiRAG(PROJECT_ID, REGION, MODEL_NAME)

        # 2. Search Confluence (with Copper filter)
        search_results = confluence.search_content(
            user_query=args.question,
            copper_filter_terms=["copper", "api"], # Add relevant filter terms
            limit=args.limit,
            space_key=args.space
            )

        if not search_results or not search_results.get("results"):
            logger.warning("No relevant pages found in Confluence for the query and 'Copper' filter.")
            print("Could not find relevant information in Confluence.")
            # Optionally, still ask Gemini without context?
            # answer = gemini.generate_answer(args.question, context=[], stream=not args.nostream)
            # print("\nGemini (without context):")
            # print(answer)
            return

        # 3. Fetch Content from Search Results
        context_pages = []
        page_ids = [result['content']['id'] for result in search_results.get('results', []) if 'content' in result and 'id' in result['content']]

        if not page_ids:
             logger.warning("Search results found, but could not extract page IDs.")
             print("Could not retrieve content details from Confluence search results.")
             return

        logger.info(f"Fetching content for top {len(page_ids)} page(s): {page_ids}")
        for page_id in page_ids:
            content = confluence.get_page_content(page_id)
            if content and content.get('text'): # Ensure content and text exist
                context_pages.append(content)
            else:
                 logger.warning(f"Skipping page ID {page_id} due to fetch error or empty content.")

        if not context_pages:
            logger.error("Failed to fetch content for any relevant Confluence pages.")
            print("Error: Unable to retrieve content from Confluence for context.")
            return

        # 4. Generate Answer using Gemini
        answer = gemini.generate_answer(
            question=args.question,
            context=context_pages,
            stream=not args.nostream
        )

        # (Streaming output is handled within generate_answer)
        if args.nostream:
            print("\nGemini Response:")
            print(answer)

        logger.info("RAG process completed.")

    except ConnectionError as e:
         logger.critical(f"Halting execution due to Confluence connection error: {e}")
         print(f"Critical Error: Could not connect to Confluence. Please check URL and credentials. Error: {e}")
         sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        print(f"An unexpected error stopped the process. Check logs. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
Use code with caution.
Python
How to Run:

Save the code above as a Python file (e.g., copper_rag.py).

Make sure you've set the environment variables mentioned in the prerequisites.

Run from your terminal:

python copper_rag.py "What are the rate limits for the Copper Order API?"
Use code with caution.
Bash
Or to search within a specific space:

python copper_rag.py "How do I authenticate with the Copper Customer API?" --space COPPERDOCS
Use code with caution.
Bash
Or to get a non-streaming response:

python copper_rag.py "Explain the different statuses for a Copper transaction" --nostream
Use code with caution.
Bash
Explanation and Key Changes:

Combined Structure: All necessary code (imports, config, Confluence client, Gemini interaction, main logic) is in one file.

Configuration: Uses os.environ.get for all sensitive details and essential parameters. Includes validation.

Confluence Client:

Takes URL, username, token on init.

Includes a basic _test_connection method called during initialization.

search_content: Modified to accept user_query and copper_filter_terms. It constructs a CQL query combining these using AND. You can adjust copper_filter_terms if needed (e.g., ["copper", "api"]). Allows optional space_key filtering. Includes error handling for network issues and JSON decoding.

get_page_content: Uses BeautifulSoup with html5lib for more robust HTML parsing to extract clean text. Returns a dictionary with id, title, url, and text. Handles errors gracefully.

GeminiRAG Class:

Initializes Vertex AI and the specified Gemini model.

Sets default safety settings and generation config (can be overridden).

generate_answer:

Takes the question and a list of context dictionaries (from get_page_content).

Builds a structured prompt including system instructions, the formatted context (with source page info), and the user question.

Includes basic context length truncation (you might need a more sophisticated token-based approach for production).

Handles both streaming and non-streaming responses.

Includes error handling for Gemini API calls.

Uses the system_instruction parameter for better control.

Main Logic (main function):

Uses argparse to get the user's question and optional arguments (space, limit, nostream).

Initializes the clients.

Calls search_content with the question and hardcoded "copper" filter terms.

Calls get_page_content for the top search results.

Filters out pages where content fetching failed or was empty.

Calls gemini.generate_answer with the question and fetched context.

Prints the final answer (streaming handled inside the class).

Includes top-level error handling for critical failures (like connection errors).

This script provides a solid foundation for your hackathon project, directly addressing your requirement to filter Confluence content for "Copper" and use Gemini to answer questions based on that filtered information. Remember to handle credentials securely and be mindful of API rate limits and costs.



























{
  "error": "Providing a complete, end-to-end, '50-year proof' RAG system codebase in a single response is impractical and goes against the principles of iterative development, testing, and adapting to evolving technologies. Building such a system requires careful planning, setup, testing, and refinement based on specific environment constraints and performance.",
  "explanation": "A production-ready system involves multiple components, infrastructure setup (like a persistent vector database, potentially task queues, deployment configurations), robust error handling, security considerations (API keys, authentication), monitoring, and comprehensive testing. Simply providing a large block of code won't result in a working or reliable system without the necessary setup and understanding.",
  "alternative_approach": "Instead of one monolithic code dump, I will provide the structured code components for the architecture described previously. This allows you to understand each part, set it up incrementally, and adapt it. I will focus on the core logic using Python, FastAPI for the API, ChromaDB for a local vector store (suitable for a hackathon), and Streamlit for a simple UI.",
  "code_components": [
    {
      "filename": "requirements.txt",
      "description": "Python dependencies required for the project.",
      "code": [
        "fastapi",
        "uvicorn[standard]",
        "requests",
        "python-dotenv",
        "google-cloud-aiplatform",
        "google-auth",
        "beautifulsoup4",
        "html5lib",
        "chromadb",
        "streamlit",
        "pydantic",
        "tiktoken"
      ]
    },
    {
      "filename": ".env.example",
      "description": "Example environment variables. Create a `.env` file and fill in your actual values.",
      "code": [
        "CONFLUENCE_URL=https://yourcompany.atlassian.net",
        "CONFLUENCE_USERNAME=your_email@example.com",
        "CONFLUENCE_API_TOKEN=your_confluence_api_token",
        "PROJECT_ID=your-gcp-project-id",
        "REGION=us-central1",
        "GEMINI_MODEL_NAME=gemini-1.5-flash-001",
        "EMBEDDING_MODEL_NAME=text-embedding-004",
        "VECTOR_DB_PATH=./chroma_db",
        "COLLECTION_NAME=confluence_copper",
        "API_HOST=0.0.0.0",
        "API_PORT=8000",
        "",
        "# Optional: For ingestion filtering",
        "CONFLUENCE_SPACE_KEY=YOUR_SPACE_KEY # If filtering by space",
        "# Optional: Adjust chunking / embedding",
        "CHUNK_SIZE=1000",
        "CHUNK_OVERLAP=150",
        "# Optional: Limit number of search results for context",
        "CONTEXT_NUM_RESULTS=5"
      ]
    },
    {
      "filename": "src/config.py",
      "description": "Loads configuration from environment variables using python-dotenv.",
      "code": [
        "import os",
        "from dotenv import load_dotenv",
        "",
        "load_dotenv()",
        "",
        "CONFLUENCE_URL = os.getenv(\"CONFLUENCE_URL\")",
        "CONFLUENCE_USERNAME = os.getenv(\"CONFLUENCE_USERNAME\")",
        "CONFLUENCE_API_TOKEN = os.getenv(\"CONFLUENCE_API_TOKEN\")",
        "PROJECT_ID = os.getenv(\"PROJECT_ID\")",
        "REGION = os.getenv(\"REGION\", \"us-central1\")",
        "GEMINI_MODEL_NAME = os.getenv(\"GEMINI_MODEL_NAME\", \"gemini-1.5-flash-001\")",
        "EMBEDDING_MODEL_NAME = os.getenv(\"EMBEDDING_MODEL_NAME\", \"text-embedding-004\")",
        "VECTOR_DB_PATH = os.getenv(\"VECTOR_DB_PATH\", \"./chroma_db\")",
        "COLLECTION_NAME = os.getenv(\"COLLECTION_NAME\", \"confluence_copper\")",
        "API_HOST = os.getenv(\"API_HOST\", \"0.0.0.0\")",
        "API_PORT = int(os.getenv(\"API_PORT\", \"8000\"))",
        "",
        "# Optional",
        "CONFLUENCE_SPACE_KEY = os.getenv(\"CONFLUENCE_SPACE_KEY\") # Optional space filter",
        "CHUNK_SIZE = int(os.getenv(\"CHUNK_SIZE\", \"1000\"))",
        "CHUNK_OVERLAP = int(os.getenv(\"CHUNK_OVERLAP\", \"150\"))",
        "CONTEXT_NUM_RESULTS = int(os.getenv(\"CONTEXT_NUM_RESULTS\", \"5\"))",
        "",
        "# Basic validation",
        "required_vars = [",
        "    CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN,",
        "    PROJECT_ID, REGION, GEMINI_MODEL_NAME, EMBEDDING_MODEL_NAME,",
        "    VECTOR_DB_PATH, COLLECTION_NAME",
        "]",
        "",
        "if not all(required_vars):",
        "    raise ValueError(\"One or more required environment variables are missing. Check .env file or environment.\")"
      ]
    },
    {
      "filename": "src/utils/text_processing.py",
      "description": "Utilities for cleaning HTML and chunking text.",
      "code": [
        "from bs4 import BeautifulSoup",
        "import tiktoken # Using tiktoken for more accurate chunking if needed",
        "import logging",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "def clean_html(html_content: str) -> str:",
        "    \"\"\"Removes scripts, styles, and extracts readable text from HTML.\"\"\"",
        "    if not html_content:",
        "        return \"\"",
        "    try:",
        "        soup = BeautifulSoup(html_content, 'html5lib')",
        "        # Remove script and style elements",
        "        for script_or_style in soup([\"script\", \"style\", \"header\", \"footer\", \"nav\"]):",
        "            script_or_style.decompose()",
        "        # Get text, attempting to preserve some structure",
        "        text = soup.get_text(separator='\\n', strip=True)",
        "        # Optional: Clean up excessive newlines",
        "        lines = (line.strip() for line in text.splitlines())",
        "        chunks = (phrase.strip() for line in lines for phrase in line.split(\"  \"))",
        "        clean_text = '\\n'.join(chunk for chunk in chunks if chunk)",
        "        return clean_text",
        "    except Exception as e:",
        "        logger.error(f\"Error cleaning HTML: {e}\", exc_info=True)",
        "        return \"\" # Return empty string on error",
        "",
        "# Simple Recursive Character Text Splitter (alternative to library)",
        "def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:",
        "    \"\"\"Splits text into chunks of specified size with overlap.\"\"\"",
        "    if not text:",
        "        return []",
        "    chunks = []",
        "    start = 0",
        "    while start < len(text):",
        "        end = start + chunk_size",
        "        chunks.append(text[start:end])",
        "        start += chunk_size - chunk_overlap",
        "        if start < 0 : start = 0 # Prevent negative start index if overlap > size",
        "    # Ensure the last chunk captures the end if overlap logic misses it",
        "    if start < len(text) and not text[start:].isspace() and len(text[start:]) > chunk_overlap // 2 :",
        "         if not chunks or text[start:] != chunks[-1][-(len(text)-start):]: # Avoid duplicate end chunks",
        "            chunks.append(text[start:])",
        "",
        "    # Filter out potentially empty or whitespace-only chunks created by overlap",
        "    return [chunk for chunk in chunks if chunk and not chunk.isspace()]",
        "",
        "# Optional: Tiktoken based chunking (more accurate for LLM context windows)",
        "def chunk_text_by_tokens(text: str, model_name: str = \"gpt-3.5-turbo\", max_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:",
        "    \"\"\"Chunks text based on token count using tiktoken.\"\"\"",
        "    if not text:",
        "        return []",
        "    try:",
        "        encoding = tiktoken.encoding_for_model(model_name)",
        "    except KeyError:",
        "        logger.warning(f\"Model {model_name} not found for tiktoken, using cl100k_base.\")",
        "        encoding = tiktoken.get_encoding(\"cl100k_base\")",
        "",
        "    tokens = encoding.encode(text)",
        "    total_tokens = len(tokens)",
        "    chunks = []",
        "    start_token = 0",
        "    while start_token < total_tokens:",
        "        end_token = start_token + max_tokens",
        "        chunk_tokens = tokens[start_token:end_token]",
        "        chunk_text = encoding.decode(chunk_tokens)",
        "        chunks.append(chunk_text.strip())",
        "        start_token += max_tokens - overlap_tokens",
        "        if start_token < 0 : start_token = 0",
        "",
        "    # Capture the tail end if missed",
        "    if start_token < total_tokens:",
        "        chunk_tokens = tokens[start_token:]",
        "        chunk_text = encoding.decode(chunk_tokens).strip()",
        "        if chunk_text and (not chunks or chunk_text != chunks[-1][-(len(chunk_text)):]):",
        "             chunks.append(chunk_text)",
        "",
        "    return [chunk for chunk in chunks if chunk and not chunk.isspace()]"
      ]
    },
    {
      "filename": "src/clients/confluence.py",
      "description": "Client for interacting with the Confluence API.",
      "code": [
        "import requests",
        "import logging",
        "from requests.auth import HTTPBasicAuth",
        "import urllib3",
        "from src.utils.text_processing import clean_html",
        "from src.config import CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN",
        "",
        "logger = logging.getLogger(__name__)",
        "urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)",
        "",
        "class ConfluenceClient:",
        "    def __init__(self):",
        "        if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):",
        "            raise ValueError(\"Confluence credentials not found in environment variables.\")",
        "        self.base_url = CONFLUENCE_URL.rstrip('/')",
        "        self.auth = HTTPBasicAuth(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)",
        "        self.api_url = f\"{self.base_url}/wiki/rest/api\"",
        "        self.headers = {\"Accept\": \"application/json\"}",
        "        logger.info(f\"ConfluenceClient initialized for {self.base_url}\")",
        "",
        "    def _make_request(self, endpoint, params=None):",
        "        url = f\"{self.api_url}/{endpoint}\"",
        "        try:",
        "            response = requests.get(",
        "                url,",
        "                headers=self.headers,",
        "                auth=self.auth,",
        "                params=params,",
        "                verify=False, # WARNING: Disables SSL verification",
        "                timeout=30",
        "            )",
        "            response.raise_for_status()",
        "            return response.json()",
        "        except requests.exceptions.RequestException as e:",
        "            logger.error(f\"Confluence API request failed for {url}. Error: {e}\")",
        "            if hasattr(e, 'response') and e.response is not None:",
        "                logger.error(f\"Status Code: {e.response.status_code}\")",
        "                logger.error(f\"Response: {e.response.text[:500]}\")",
        "            return None",
        "        except Exception as e:",
        "             logger.error(f\"Unexpected error during Confluence request to {url}: {e}\", exc_info=True)",
        "             return None",
        "",
        "    def search_pages(self, query_terms: list[str], space_key: str = None, content_type: str = \"page\", limit: int = 100):",
        "        \"\"\"Search Confluence pages using CQL.\"\"\"",
        "        cql_parts = [f'text ~ \"{term}\"' for term in query_terms]",
        "        cql_parts.append(f'type=\"{content_type}\"')",
        "        if space_key:",
        "            cql_parts.append(f'space=\"{space_key}\"')",
        "        cql = \" AND \".join(cql_parts)",
        "        logger.info(f\"Searching Confluence with CQL: {cql}, Limit: {limit}\")",
        "        all_results = []",
        "        start = 0",
        "        while True:",
        "            params = {\"cql\": cql, \"limit\": limit, \"start\": start, \"expand\": \"space,version\"}",
        "            data = self._make_request(\"content/search\", params=params)",
        "            if not data or 'results' not in data:",
        "                break",
        "            results = data.get('results', [])",
        "            all_results.extend(results)",
        "            logger.info(f\"Fetched {len(results)} results, total so far: {len(all_results)}\")",
        "            if len(results) < limit:", # Check if this is the last page
        "                 break",
        "            start += limit # Prepare for the next page",
        "            # Safety break to prevent infinite loops in case API behavior changes",
        "            if start > 1000: # Limit to roughly 1000 results total",
        "                 logger.warning(\"Reached maximum search result limit (1000). Stopping search.\")",
        "                 break",
        "        logger.info(f\"Total pages found matching query: {len(all_results)}\")",
        "        return all_results",
        "",
        "    def get_page_details(self, page_id: str):",
        "        \"\"\"Get page details including body content.\"\"\"",
        "        logger.debug(f\"Fetching details for page ID: {page_id}\")",
        "        params = {\"expand\": \"body.storage,version,space\"}",
        "        data = self._make_request(f\"content/{page_id}\", params=params)",
        "        if not data:",
        "            return None",
        "",
        "        html_content = data.get(\"body\", {}).get(\"storage\", {}).get(\"value\", \"\")",
        "        clean_text = clean_html(html_content)",
        "        page_title = data.get(\"title\", \"Unknown Title\")",
        "        space_key = data.get(\"space\", {}).get(\"key\", \"UNK\")",
        "        version = data.get(\"version\", {}).get(\"number\", 0)",
        "        last_modified = data.get(\"version\", {}).get(\"when\", \"\")",
        "        page_url = f\"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}\"",
        "",
        "        if not clean_text:",
        "             logger.warning(f\"Page ID {page_id} ('{page_title}') had no processable content after cleaning.\")",
        "",
        "        return {",
        "            \"id\": page_id,",
        "            \"title\": page_title,",
        "            \"url\": page_url,",
        "            \"space_key\": space_key,",
        "            \"version\": version,",
        "            \"last_modified\": last_modified,",
        "            \"text\": clean_text",
        "        }",
        ""
      ]
    },
    {
      "filename": "src/clients/vector_db.py",
      "description": "Client for interacting with the ChromaDB vector database.",
      "code": [
        "import chromadb",
        "from chromadb.utils import embedding_functions",
        "import logging",
        "from typing import List, Dict, Optional",
        "from src.config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, PROJECT_ID, REGION",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "class VectorDBClient:",
        "    def __init__(self):",
        "        try:",
        "            self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)",
        "            # Use Vertex AI Embedding Function directly if possible (check ChromaDB docs)",
        "            # For simplicity here, we might embed outside and pass embeddings in,",
        "            # or use a sentence-transformer model if Vertex AI integration isn't straightforward.",
        "            # Example using Vertex AI Embeddings (requires google-cloud-aiplatform)",
        "            self.embedding_function = embedding_functions.GoogleVertexEmbeddingFunction(",
        "                 api_key=None, # Handled by application default credentials",
        "                 project_id=PROJECT_ID,",
        "                 location=REGION,",
        "                 model_name=EMBEDDING_MODEL_NAME",
        "            )",
        "            # Use get_or_create_collection",
        "            self.collection = self.client.get_or_create_collection(",
        "                name=COLLECTION_NAME,",
        "                embedding_function=self.embedding_function,",
        "                metadata={\"hnsw:space\": \"cosine\"} # Specify distance metric if needed",
        "            )",
        "            logger.info(f\"ChromaDB client initialized. Collection '{COLLECTION_NAME}' loaded/created.\")",
        "            logger.info(f\"Collection size: {self.collection.count()} documents\")",
        "        except Exception as e:",
        "            logger.error(f\"Failed to initialize ChromaDB client: {e}\", exc_info=True)",
        "            raise",
        "",
        "    def upsert_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):",
        "        \"\"\"Adds or updates documents in the collection.\"\"\"",
        "        if not documents or not metadatas or not ids:",
        "            logger.warning(\"Upsert attempt with empty documents, metadatas, or ids.\")",
        "            return",
        "        if not (len(documents) == len(metadatas) == len(ids)):",
        "             logger.error(\"Mismatch in lengths of documents, metadatas, and ids for upsert.\")",
        "             raise ValueError(\"Documents, metadatas, and ids must have the same length.\")",
        "        try:",
        "            # ChromaDB handles embedding generation via the collection's embedding function",
        "            self.collection.upsert(",
        "                documents=documents,",
        "                metadatas=metadatas,",
        "                ids=ids",
        "            )",
        "            logger.info(f\"Upserted {len(ids)} documents into collection '{COLLECTION_NAME}'.\")",
        "        except Exception as e:",
        "            logger.error(f\"Failed to upsert documents: {e}\", exc_info=True)",
        "            # Consider partial failure handling if needed",
        "",
        "    def query_collection(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict] = None) -> Optional[Dict]:",
        "        \"\"\"Queries the collection for similar documents.\"\"\"",
        "        try:",
        "            results = self.collection.query(",
        "                query_texts=query_texts,",
        "                n_results=n_results,",
        "                where=where, # Optional metadata filtering",
        "                include=['metadatas', 'documents', 'distances'] # Include necessary fields",
        "            )",
        "            logger.info(f\"Query returned results for {len(query_texts)} query texts.\")",
        "            return results",
        "        except Exception as e:",
        "            logger.error(f\"Failed to query collection: {e}\", exc_info=True)",
        "            return None",
        "",
        "    def get_collection_count(self):",
        "        \"\"\"Returns the number of items in the collection.\"\"\"",
        "        try:",
        "             return self.collection.count()",
        "        except Exception as e:",
        "             logger.error(f\"Failed to get collection count: {e}\", exc_info=True)",
        "             return 0"
      ]
    },
     {
        "filename": "src/clients/llm.py",
        "description": "Client for interacting with the Gemini LLM via Vertex AI.",
        "code": [
            "import vertexai",
            "from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, SafetySetting, HarmCategory",
            "import logging",
            "from typing import List, Dict, Optional",
            "from src.config import PROJECT_ID, REGION, GEMINI_MODEL_NAME",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            "class GeminiClient:",
            "    def __init__(self):",
            "        try:",
            "            vertexai.init(project=PROJECT_ID, location=REGION)",
            "            self.model = GenerativeModel(GEMINI_MODEL_NAME)",
            "            logger.info(f\"Initialized Gemini model: {GEMINI_MODEL_NAME} in {PROJECT_ID}/{REGION}\")",
            "            # Define default safety settings (adjust as needed)",
            "            self.safety_settings = {",
            "                HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,",
            "                HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,",
            "                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,",
            "                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,",
            "            }",
            "            # Default Generation Config (can be overridden)",
            "            self.default_generation_config = GenerationConfig(",
            "                temperature=0.3,",
            "                top_p=0.9,",
            "                max_output_tokens=4096",
            "            )",
            "        except Exception as e:",
            "            logger.error(f\"Failed to initialize Vertex AI or Gemini Model: {e}\", exc_info=True)",
            "            raise",
            "",
            "    def generate_response(self, prompt_parts: List[Part], system_instruction: Optional[str] = None, stream=False, generation_config: Optional[GenerationConfig] = None):",
            "        \"\"\"Generates a response from the Gemini model.\"\"\"",
            "        gen_config = generation_config if generation_config else self.default_generation_config",
            "        try:",
            "            logger.info(f\"Sending request to Gemini. Streaming: {stream}\")",
            "            response = self.model.generate_content(",
            "                prompt_parts,",
            "                generation_config=gen_config,",
            "                safety_settings=self.safety_settings,",
            "                stream=stream,",
            "                system_instruction=system_instruction",
            "            )",
            "            return response",
            "        except Exception as e:",
            "            logger.error(f\"Error generating response from Gemini: {e}\", exc_info=True)",
            "            return None # Or raise a custom exception"
        ]
    },
    {
      "filename": "src/core/rag.py",
      "description": "Core RAG orchestration logic.",
      "code": [
        "import logging",
        "from typing import List, Dict, Optional",
        "from src.clients.vector_db import VectorDBClient",
        "from src.clients.llm import GeminiClient",
        "from src.config import CONTEXT_NUM_RESULTS",
        "from vertexai.generative_models import Part",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "class RAGOrchestrator:",
        "    def __init__(self, vector_db_client: VectorDBClient, llm_client: GeminiClient):",
        "        self.vector_db = vector_db_client",
        "        self.llm = llm_client",
        "",
        "    def generate_rag_response(self, question: str, stream: bool = False) -> str:",
        "        \"\"\"Orchestrates the RAG process: query -> search -> context -> prompt -> LLM -> response.\"\"\"",
        "        logger.info(f\"Processing RAG request for question: '{question[:50]}...'\" )",
        "",
        "        # 1. Query Vector Database for relevant context",
        "        try:",
        "            search_results = self.vector_db.query_collection(",
        "                query_texts=[question],",
        "                n_results=CONTEXT_NUM_RESULTS",
        "            )",
        "        except Exception as e:",
        "             logger.error(f\"Vector DB query failed: {e}\", exc_info=True)",
        "             return \"Error: Could not retrieve context from the knowledge base.\"",
        "",
        "        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:",
        "            logger.warning(\"No relevant context found in Vector DB for the question.\")",
        "            # Optional: Fallback - ask LLM without context or return specific message",
        "            context_str = \"No relevant context found in the knowledge base.\"",
        "            # return \"I couldn't find relevant information in the knowledge base to answer your question.\" ",
        "        else:",
        "            # 2. Format context",
        "            context_docs = search_results['documents'][0]",
        "            metadatas = search_results['metadatas'][0]",
        "            # distances = search_results['distances'][0] # Optional: use distances for logging/filtering",
        "            context_parts = []",
        "            for i, doc in enumerate(context_docs):",
        "                meta = metadatas[i]",
        "                title = meta.get('title', 'N/A')",
        "                url = meta.get('url', 'N/A')",
        "                # dist = distances[i] # Optional",
        "                context_parts.append(f\"Source {i+1} (Title: {title}, URL: {url}):\\n{doc}\")",
        "            context_str = \"\\n\\n---\\n\\n\".join(context_parts)",
        "            logger.info(f\"Retrieved {len(context_docs)} context documents.\")",
        "",
        "        # 3. Construct Prompt for LLM",
        "        system_instruction = (",
        "            \"You are a helpful assistant knowledgeable about Copper APIs based on internal Confluence documentation. \"",
        "            \"Answer the user's question using *only* the provided context from Confluence pages below. \"",
        "            \"If the context doesn't contain the answer, clearly state that the information wasn't found in the provided documents. \"",
        "            \"Cite the source title or URL when possible. Be concise and accurate.\"",
        "        )",
        "",
        "        # Limit context size (simple character limit, improve with token counting if needed)",
        "        MAX_CONTEXT_CHARS = 28000 # Adjust based on model token limit",
        "        if len(context_str) > MAX_CONTEXT_CHARS:",
        "            logger.warning(f\"Context length ({len(context_str)} chars) exceeds limit. Truncating.\")",
        "            context_str = context_str[:MAX_CONTEXT_CHARS] + \"\\n... [Context Truncated]\"",
        "",
        "        prompt_parts = [",
        "            Part.from_text(f\"Context from Confluence Documents:\\n{context_str}\"),",
        "            Part.from_text(f\"\\n\\nUser Question: {question}\\n\\nAnswer:\")",
        "        ]",
        "",
        "        # 4. Call LLM",
        "        llm_response = self.llm.generate_response(",
        "            prompt_parts=prompt_parts,",
        "            system_instruction=system_instruction,",
        "            stream=stream",
        "        )",
        "",
        "        if llm_response is None:",
        "            return \"Error: Failed to get a response from the language model.\"",
        "",
        "        # 5. Process and return response",
        "        if stream:",
        "            # For streaming, the API route would handle iterating the generator",
        "            # This function might return the generator itself in a real scenario",
        "            # For simplicity here, let's consume it (won't actually stream to caller)",
        "            logger.info(\"Consuming stream for non-streaming return (API should handle differently)\")",
        "            full_response_text = \"\"",
        "            try:",
        "                 for chunk in llm_response:",
        "                    if chunk.candidates and chunk.candidates[0].content.parts:",
        "                        full_response_text += chunk.candidates[0].content.parts[0].text",
        "            except Exception as e:",
        "                 logger.error(f\"Error consuming LLM stream: {e}\")",
        "                 return \"Error processing streamed response from LLM.\"",
        "            logger.info(f\"RAG Process completed. Response length: {len(full_response_text)}\")",
        "            return full_response_text",
        "        else:",
        "            # Handle non-streaming response",
        "            try:",
        "                 if llm_response.candidates and llm_response.candidates[0].content.parts:",
        "                     full_response_text = llm_response.candidates[0].content.parts[0].text",
        "                     logger.info(f\"RAG Process completed. Response length: {len(full_response_text)}\")",
        "                     return full_response_text",
        "                 else:",
        "                     finish_reason = llm_response.candidates[0].finish_reason if llm_response.candidates else 'N/A'",
        "                     logger.warning(f\"LLM response was empty or potentially blocked. Finish Reason: {finish_reason}\")",
        "                     safety_feedback = llm_response.prompt_feedback if hasattr(llm_response, 'prompt_feedback') else 'N/A'",
        "                     logger.warning(f\"Safety Feedback: {safety_feedback}\")",
        "                     return \"Error: The language model did not return a valid response. It might have been blocked.\"",
        "            except Exception as e:",
        "                 logger.error(f\"Error processing non-streamed LLM response: {e}\", exc_info=True)",
        "                 return \"Error processing response from LLM.\""
        "        ",
        ""
      ]
    },
    {
      "filename": "src/ingestion/main.py",
      "description": "Script to run the data ingestion pipeline.",
      "code": [
        "import logging",
        "import time",
        "from src.clients.confluence import ConfluenceClient",
        "from src.clients.vector_db import VectorDBClient",
        "from src.utils.text_processing import chunk_text",
        "from src.config import (",
        "    CONFLUENCE_SPACE_KEY, CHUNK_SIZE, CHUNK_OVERLAP,",
        "    COLLECTION_NAME # Used for logging",
        ")",
        "",
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')",
        "logger = logging.getLogger(__name__)",
        "",
        "def run_ingestion():",
        "    logger.info(\"Starting Confluence data ingestion pipeline...\")",
        "    start_time = time.time()",
        "",
        "    try:",
        "        confluence_client = ConfluenceClient()",
        "        vector_db_client = VectorDBClient()",
        "    except Exception as e:",
        "        logger.critical(f\"Failed to initialize clients: {e}. Aborting ingestion.\", exc_info=True)",
        "        return",
        "",
        "    # 1. Search for relevant pages (Add your Copper-specific terms here)",
        "    # Example: Search for pages containing 'copper' and 'api' in a specific space",
        "    query_terms = [\"copper\", \"api\"] # Modify as needed",
        "    logger.info(f\"Searching for pages with terms: {query_terms} in space: {CONFLUENCE_SPACE_KEY or 'ALL'}\")",
        "    pages = confluence_client.search_pages(query_terms=query_terms, space_key=CONFLUENCE_SPACE_KEY, limit=50) # Adjust limit",
        "",
        "    if not pages:",
        "        logger.warning(\"No pages found matching the search criteria. Ingestion finished.\")",
        "        return",
        "",
        "    logger.info(f\"Found {len(pages)} potentially relevant pages.\")",
        "",
        "    all_chunks = []",
        "    all_metadatas = []",
        "    all_ids = []",
        "    processed_page_count = 0",
        "",
        "    # 2. Fetch, Clean, Chunk, and Prepare for Upsert",
        "    for page_summary in pages:",
        "        page_id = page_summary.get('id') or page_summary.get('content', {}).get('id')",
        "        if not page_id:",
        "            logger.warning(f\"Skipping search result missing page ID: {page_summary.get('title', 'N/A')}\")",
        "            continue",
        "",
        "        page_details = confluence_client.get_page_details(page_id)",
        "        if not page_details or not page_details.get('text'):",
        "            logger.warning(f\"Skipping page ID {page_id} due to fetch error or empty content.\")",
        "            continue",
        "",
        "        logger.info(f\"Processing page: '{page_details['title']}' (ID: {page_id}) - {len(page_details['text'])} chars\")",
        "        text = page_details['text']",
        "        page_title = page_details['title']",
        "        page_url = page_details['url']",
        "        space_key = page_details['space_key']",
        "        last_modified = page_details['last_modified']",
        "        version = page_details['version']",
        "",
        "        # Chunk the text",
        "        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)",
        "        # Alternative: Use token chunking",
        "        # chunks = chunk_text_by_tokens(text, max_tokens=CHUNK_SIZE, overlap_tokens=CHUNK_OVERLAP)",
        "",
        "        if not chunks:",
        "             logger.warning(f\"No chunks generated for page ID {page_id} ('{page_title}')\")",
        "             continue",
        "",
        "        logger.info(f\"Generated {len(chunks)} chunks for page ID {page_id}.\")",
        "",
        "        for i, chunk in enumerate(chunks):",
        "            chunk_id = f\"{page_id}_chunk_{i}\"",
        "            metadata = {",
        "                \"page_id\": page_id,",
        "                \"title\": page_title,",
        "                \"url\": page_url,",
        "                \"space_key\": space_key,",
        "                \"chunk_index\": i,",
        "                \"last_modified\": last_modified,",
        "                \"version\": version",
        "                # Add other relevant metadata if needed",
        "            }",
        "            all_chunks.append(chunk)",
        "            all_metadatas.append(metadata)",
        "            all_ids.append(chunk_id)",
        "        processed_page_count += 1",
        "",
        "    # 3. Upsert into Vector DB (in batches if necessary, ChromaDB handles batching internally to some extent)",
        "    if all_chunks:",
        "        logger.info(f\"Preparing to upsert {len(all_chunks)} total chunks from {processed_page_count} pages into collection '{COLLECTION_NAME}'.\")",
        "        try:",
        "            vector_db_client.upsert_documents(",
        "                documents=all_chunks,",
        "                metadatas=all_metadatas,",
        "                ids=all_ids",
        "            )",
        "            logger.info(\"Upsert operation completed.\")",
        "            logger.info(f\"Vector DB now contains approximately {vector_db_client.get_collection_count()} documents.\")",
        "        except Exception as e:",
        "             logger.error(f\"Failed during vector DB upsert: {e}\", exc_info=True)",
        "    else:",
        "        logger.warning(\"No valid chunks were generated to upsert.\")",
        "",
        "    end_time = time.time()",
        "    logger.info(f\"Ingestion pipeline finished in {end_time - start_time:.2f} seconds.\")",
        "",
        "if __name__ == \"__main__\":",
        "    run_ingestion()",
        ""
      ]
    },
    {
      "filename": "src/api/models.py",
      "description": "Pydantic models for API request and response validation.",
      "code": [
        "from pydantic import BaseModel",
        "from typing import Optional",
        "",
        "class QueryRequest(BaseModel):",
        "    question: str",
        "    stream: Optional[bool] = False # Default to non-streaming for simple API",
        "",
        "class QueryResponse(BaseModel):",
        "    answer: str",
        "    # Optional: Add context sources if needed for debugging/display",
        "    # sources: Optional[list[dict]] = None"
      ]
    },
    {
      "filename": "src/api/main.py",
      "description": "FastAPI application entry point.",
      "code": [
        "from fastapi import FastAPI, HTTPException, Depends",
        "from fastapi.responses import StreamingResponse",
        "import logging",
        "import uvicorn",
        "from contextlib import asynccontextmanager",
        "",
        "from src.api.models import QueryRequest, QueryResponse",
        "from src.clients.vector_db import VectorDBClient",
        "from src.clients.llm import GeminiClient",
        "from src.core.rag import RAGOrchestrator",
        "from src.config import API_HOST, API_PORT",
        "",
        "# Configure logging for API",
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')",
        "logger = logging.getLogger(\"api\")",
        "",
        "# Global variables for clients (initialize lifespan)",
        "rag_orchestrator = None",
        "",
        "# Lifespan context manager for resource initialization/cleanup",
        "@asynccontextmanager",
        "async def lifespan(app: FastAPI):",
        "    global rag_orchestrator",
        "    logger.info(\"API starting up...\")",
        "    try:",
        "        vector_db = VectorDBClient()",
        "        llm_client = GeminiClient()",
        "        rag_orchestrator = RAGOrchestrator(vector_db, llm_client)",
        "        logger.info(\"RAG Orchestrator initialized.\")",
        "    except Exception as e:",
        "        logger.critical(f\"Failed to initialize resources during startup: {e}\", exc_info=True)",
        "        # Optional: Prevent FastAPI from starting if critical resources fail",
        "        raise RuntimeError(\"Critical resource initialization failed.\") from e",
        "    yield",
        "    # Cleanup (if necessary)",
        "    logger.info(\"API shutting down...\")",
        "",
        "app = FastAPI(title=\"Confluence RAG API\", lifespan=lifespan)",
        "",
        "# Dependency function to get the RAG orchestrator",
        "def get_orchestrator():",
        "    if rag_orchestrator is None:",
        "         # This should ideally not happen if lifespan manager works correctly",
        "         logger.error(\"RAG Orchestrator not initialized!\")",
        "         raise HTTPException(status_code=500, detail=\"Internal server error: RAG system not ready.\")",
        "    return rag_orchestrator",
        "",
        "@app.post(\"/query\", response_model=QueryResponse)",
        "async def ask_question(query: QueryRequest, orchestrator: RAGOrchestrator = Depends(get_orchestrator)):",
        "    \"\"\"Receives a question and returns an answer generated via RAG.\"\"\"",
        "    logger.info(f\"Received query: '{query.question[:50]}...', Stream: {query.stream}\")",
        "    if query.stream:",
        "         # Streaming is handled differently with FastAPI, returning StreamingResponse",
        "         # For simplicity in this example, we'll call the non-streaming RAG core.",
        "         # To implement actual streaming, the RAG core and LLM client need adjustments.",
        "         logger.warning(\"Streaming requested, but API endpoint currently uses non-streaming RAG core for simplicity.\")",
        "",
        "    try:",
        "        answer = orchestrator.generate_rag_response(query.question, stream=False) # Force non-stream for this endpoint",
        "        if answer.startswith(\"Error:\"):",
        "             # Distinguish between errors finding context vs. errors from LLM",
        "             if \"Could not retrieve context\" in answer:",
        "                  raise HTTPException(status_code=404, detail=answer)",
        "             else:",
        "                  raise HTTPException(status_code=500, detail=answer)",
        "        return QueryResponse(answer=answer)",
        "    except HTTPException as http_exc:",
        "         raise http_exc # Re-raise HTTP exceptions",
        "    except Exception as e:",
        "        logger.error(f\"Error processing query '{query.question[:50]}...': {e}\", exc_info=True)",
        "        raise HTTPException(status_code=500, detail=\"Internal server error while processing the question.\")",
        "",
        "@app.get(\"/health\")",
        "async def health_check():",
        "    \"\"\"Simple health check endpoint.\"\"\"",
        "    # Add checks for Vector DB and LLM client if needed",
        "    return {\"status\": \"ok\"}",
        "",
        "# --- Code to run the API directly (for development) ---",
        "# In production, use a proper ASGI server like uvicorn or hypercorn",
        "if __name__ == \"__main__\":",
        "    logger.info(f\"Starting API server on {API_HOST}:{API_PORT}\")",
        "    uvicorn.run(\"src.api.main:app\", host=API_HOST, port=API_PORT, reload=True)",
        "    # Note: `reload=True` is for development only!"
      ]
    },
    {
      "filename": "frontend/app.py",
      "description": "Simple Streamlit frontend to interact with the API.",
      "code": [
        "import streamlit as st",
        "import requests",
        "import os",
        "from dotenv import load_dotenv",
        "",
        "# Load API URL from .env or default",
        "load_dotenv(\"../.env\") # Load .env from parent directory if running from frontend/",
        "API_HOST = os.getenv(\"API_HOST\", \"localhost\") # Default to localhost if not set",
        "API_PORT = int(os.getenv(\"API_PORT\", \"8000\"))",
        "API_URL = f\"http://{API_HOST}:{API_PORT}/query\"",
        "",
        "st.set_page_config(page_title=\"Copper Confluence QA\", layout=\"wide\")",
        "st.title(\"Copper Confluence Question Answering\")",
        "st.caption(\"Ask questions about Copper APIs using information from Confluence\")",
        "",
        "# Initialize chat history",
        "if \"messages\" not in st.session_state:",
        "    st.session_state.messages = []",
        "",
        "# Display chat messages from history on app rerun",
        "for message in st.session_state.messages:",
        "    with st.chat_message(message[\"role\"]):",
        "        st.markdown(message[\"content\"])",
        "",
        "# React to user input",
        "if prompt := st.chat_input(\"Ask about Copper APIs...\"):",
        "    # Display user message in chat message container",
        "    st.chat_message(\"user\").markdown(prompt)",
        "    # Add user message to chat history",
        "    st.session_state.messages.append({\"role\": \"user\", \"content\": prompt})",
        "",
        "    # Call the backend API",
        "    try:",
        "        response = requests.post(API_URL, json={\"question\": prompt, \"stream\": False})",
        "        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)",
        "",
        "        api_response = response.json()",
        "        answer = api_response.get(\"answer\", \"Error: No answer received from API.\")",
        "",
        "        # Display assistant response in chat message container",
        "        with st.chat_message(\"assistant\"):",
        "            st.markdown(answer)",
        "        # Add assistant response to chat history",
        "        st.session_state.messages.append({\"role\": \"assistant\", \"content\": answer})",
        "",
        "    except requests.exceptions.ConnectionError:",
        "         error_msg = f\"Connection Error: Could not connect to the backend API at {API_URL}. Is it running?\"",
        "         st.error(error_msg)",
        "         st.session_state.messages.append({\"role\": \"assistant\", \"content\": error_msg})",
        "    except requests.exceptions.RequestException as e:",
        "        error_detail = \"No detail provided.\"",
        "        if e.response is not None:",
        "            try:",
        "                 error_detail = e.response.json().get('detail', e.response.text)",
        "            except ValueError:",
        "                 error_detail = e.response.text",
        "        error_msg = f\"API Error: {e.response.status_code if e.response is not None else 'N/A'} - {error_detail}\"",
        "        st.error(error_msg)",
        "        st.session_state.messages.append({\"role\": \"assistant\", \"content\": error_msg})",
        "    except Exception as e:",
        "        error_msg = f\"An unexpected error occurred: {e}\"",
        "        st.error(error_msg)",
        "        st.session_state.messages.append({\"role\": \"assistant\", \"content\": error_msg})"
      ]
    }
  ],
  "setup_and_run_instructions": [
    "**1. Project Structure:** Create directories (`src`, `src/api`, `src/clients`, `src/core`, `src/ingestion`, `src/utils`, `frontend`, `chroma_db`). Place the corresponding Python files into these directories.",
    "**2. Environment:** Create a `.env` file in the project's root directory. Copy the contents of `.env.example` into it and fill in your actual Confluence, GCP/Vertex AI credentials, and desired settings.",
    "**3. Dependencies:** Create a virtual environment and install requirements:",
    "   ```bash",
    "   python -m venv venv",
    "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
    "   pip install -r requirements.txt",
    "   ```",
    "**4. Google Cloud Authentication:** Ensure your environment is authenticated to GCP (usually via `gcloud auth application-default login`).",
    "**5. Run Ingestion:** Execute the ingestion script first to populate the vector database:",
    "   ```bash",
    "   python -m src.ingestion.main",
    "   ```",
    "   *Note: This might take time depending on the number of pages.* Check the logs for progress and potential errors.",
    "**6. Run Backend API:** Start the FastAPI server:",
    "   ```bash",
    "   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload",
    "   ```",
    "   *Note: `--reload` is for development. Use a proper ASGI server like Gunicorn in production.*",
    "**7. Run Frontend:** Open a *new* terminal, activate the virtual environment again, and run the Streamlit app:",
    "   ```bash",
    "   streamlit run frontend/app.py",
    "   ```",
    "**8. Interact:** Open the URL provided by Streamlit in your browser and ask questions.",
    "**Important:** This setup uses a *local, persistent* ChromaDB stored in the `./chroma_db` directory. For production or team collaboration, consider a managed vector database or a centrally accessible ChromaDB instance."
  ],
  "final_disclaimer": "This code provides a structural foundation based on your requirements. It is essential to test, debug, refine, and adapt this code for your specific hackathon environment, Confluence content structure, and performance needs. It is **not** production-ready '50-year proof' code but a comprehensive starting point built on sound RAG principles."
}
























{
  "error": "Providing the entire backend (ingestion, vector database interaction, RAG logic, API server) as a single, monolithic, immediately runnable Python file is highly discouraged for real-world applications due to maintainability, testability, and scalability issues. However, understanding you want a consolidated starting point to see the pieces working together before refactoring, I will provide a single file containing the core backend logic. Please note this is **not production-ready** and should be broken down into modules as discussed previously.",
  "consolidated_backend_code": {
    "filename": "unified_backend.py",
    "description": "A single Python file combining Confluence interaction, text processing, ChromaDB vector store, Gemini embedding/generation, RAG logic, and a FastAPI API. This is for demonstration and should be refactored.",
    "code": [
      "#!/usr/bin/env python3",
      "",
      "import logging",
      "import os",
      "import sys",
      "import time",
      "import json",
      "from contextlib import asynccontextmanager",
      "",
      "# --- Core Libraries ---",
      "import requests",
      "from requests.auth import HTTPBasicAuth",
      "import urllib3",
      "from bs4 import BeautifulSoup",
      "import chromadb",
      "from chromadb.utils import embedding_functions",
      "import uvicorn",
      "from fastapi import FastAPI, HTTPException, Depends",
      "from pydantic import BaseModel",
      "from dotenv import load_dotenv",
      "",
      "# --- Google Cloud / Vertex AI ---",
      "import vertexai",
      "from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, SafetySetting, HarmCategory",
      "from vertexai.language_models import TextEmbeddingModel # Separate import for embedding model",
      "",
      "# --- Configuration ---",
      "load_dotenv() # Load environment variables from .env file",
      "",
      "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')",
      "logger = logging.getLogger(\"unified_backend\")",
      "urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # Suppress insecure request warnings",
      "",
      "# --- Load Config Values ---",
      "CONFLUENCE_URL = os.getenv(\"CONFLUENCE_URL\")",
      "CONFLUENCE_USERNAME = os.getenv(\"CONFLUENCE_USERNAME\")",
      "CONFLUENCE_API_TOKEN = os.getenv(\"CONFLUENCE_API_TOKEN\")",
      "PROJECT_ID = os.getenv(\"PROJECT_ID\")",
      "REGION = os.getenv(\"REGION\", \"us-central1\")",
      "GEMINI_MODEL_NAME = os.getenv(\"GEMINI_MODEL_NAME\", \"gemini-1.5-flash-001\")",
      "EMBEDDING_MODEL_NAME = os.getenv(\"EMBEDDING_MODEL_NAME\", \"text-embedding-004\")",
      "VECTOR_DB_PATH = os.getenv(\"VECTOR_DB_PATH\", \"./chroma_db_unified\") # Use different path",
      "COLLECTION_NAME = os.getenv(\"COLLECTION_NAME\", \"confluence_copper_unified\")",
      "API_HOST = os.getenv(\"API_HOST\", \"0.0.0.0\")",
      "API_PORT = int(os.getenv(\"API_PORT\", \"8000\"))",
      "CONFLUENCE_SPACE_KEY = os.getenv(\"CONFLUENCE_SPACE_KEY\")",
      "CHUNK_SIZE = int(os.getenv(\"CHUNK_SIZE\", \"1000\"))",
      "CHUNK_OVERLAP = int(os.getenv(\"CHUNK_OVERLAP\", \"150\"))",
      "CONTEXT_NUM_RESULTS = int(os.getenv(\"CONTEXT_NUM_RESULTS\", \"5\"))",
      "INGESTION_QUERY_TERMS = os.getenv(\"INGESTION_QUERY_TERMS\", \"copper,api\").split(',') # Comma-separated terms",
      "",
      "# --- Basic Config Validation ---",
      "required_vars = [CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, PROJECT_ID, REGION, GEMINI_MODEL_NAME, EMBEDDING_MODEL_NAME, VECTOR_DB_PATH, COLLECTION_NAME]",
      "if not all(required_vars):",
      "    logger.critical(\"Missing required environment variables. Check .env file or environment.\")",
      "    sys.exit(1)",
      "",
      "# =========================================",
      "# === Text Processing Utilities ===",
      "# =========================================",
      "def clean_html(html_content: str) -> str:",
      "    # (Same implementation as in src/utils/text_processing.py)",
      "    if not html_content: return \"\"",
      "    try:",
      "        soup = BeautifulSoup(html_content, 'html5lib')",
      "        for script_or_style in soup([\"script\", \"style\", \"header\", \"footer\", \"nav\"]): script_or_style.decompose()",
      "        text = soup.get_text(separator='\\n', strip=True)",
      "        lines = (line.strip() for line in text.splitlines())",
      "        chunks = (phrase.strip() for line in lines for phrase in line.split(\"  \"))",
      "        return '\\n'.join(chunk for chunk in chunks if chunk)",
      "    except Exception as e: logger.error(f\"Error cleaning HTML: {e}\"); return \"\"",
      "",
      "def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:",
      "    # (Same implementation as in src/utils/text_processing.py)",
      "    if not text: return []",
      "    chunks = []",
      "    start = 0",
      "    while start < len(text):",
      "        end = start + chunk_size",
      "        chunks.append(text[start:end])",
      "        start += chunk_size - chunk_overlap",
      "        if start < 0 : start = 0",
      "    if start < len(text) and not text[start:].isspace() and len(text[start:]) > chunk_overlap // 2 :",
      "         if not chunks or text[start:] != chunks[-1][-(len(text)-start):]:",
      "            chunks.append(text[start:])",
      "    return [chunk for chunk in chunks if chunk and not chunk.isspace()]",
      "",
      "# =========================================",
      "# === Global Client Instances (Initialize in lifespan) ===",
      "# =========================================",
      "confluence_client = None",
      "vector_db_collection = None",
      "llm_client_gemini = None",
      "embedding_client_vertex = None",
      "",
      "# =========================================",
      "# === Confluence Client Logic ===",
      "# =========================================",
      "class _ConfluenceClient:",
      "    # (Combine relevant methods from src/clients/confluence.py here)",
      "    def __init__(self):",
      "        self.base_url = CONFLUENCE_URL.rstrip('/')",
      "        self.auth = HTTPBasicAuth(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)",
      "        self.api_url = f\"{self.base_url}/wiki/rest/api\"",
      "        self.headers = {\"Accept\": \"application/json\"}",
      "        logger.info(f\"_ConfluenceClient initialized for {self.base_url}\")",
      "",
      "    def _make_request(self, endpoint, params=None):",
      "        url = f\"{self.api_url}/{endpoint}\"",
      "        try:",
      "            response = requests.get(url, headers=self.headers, auth=self.auth, params=params, verify=False, timeout=30)",
      "            response.raise_for_status()",
      "            return response.json()",
      "        except requests.exceptions.RequestException as e:",
      "            logger.error(f\"Confluence API request failed for {url}. Error: {e}\")",
      "            if hasattr(e, 'response') and e.response is not None: logger.error(f\"Status: {e.response.status_code}, Resp: {e.response.text[:200]}\")",
      "            return None",
      "",
      "    def search_pages(self, query_terms: list[str], space_key: str = None, content_type: str = \"page\", limit_per_req: int = 50, max_total: int = 1000):",
      "        cql_parts = [f'text ~ \"{term.strip()}\"' for term in query_terms if term.strip()]",
      "        if not cql_parts: logger.warning(\"No valid query terms for Confluence search.\"); return []",
      "        cql_parts.append(f'type=\"{content_type}\"')",
      "        if space_key: cql_parts.append(f'space=\"{space_key}\"')",
      "        cql = \" AND \".join(cql_parts)",
      "        logger.info(f\"Searching Confluence with CQL: {cql}, Max Results: {max_total}\")",
      "        all_results = []",
      "        start = 0",
      "        while True:",
      "            if len(all_results) >= max_total: break",
      "            current_limit = min(limit_per_req, max_total - len(all_results))",
      "            params = {\"cql\": cql, \"limit\": current_limit, \"start\": start, \"expand\": \"space,version\"}",
      "            data = self._make_request(\"content/search\", params=params)",
      "            if not data or 'results' not in data:",
      "                break",
      "            results = data.get('results', [])",
      "            all_results.extend(results)",
      "            logger.info(f\"Fetched {len(results)} results (start={start}), total so far: {len(all_results)}\")",
      "            if len(results) < current_limit: break",
      "            start += current_limit",
      "        logger.info(f\"Total pages found matching query: {len(all_results)}\")",
      "        return all_results",
      "",
      "    def get_page_details(self, page_id: str):",
      "        logger.debug(f\"Fetching details for page ID: {page_id}\")",
      "        params = {\"expand\": \"body.storage,version,space\"}",
      "        data = self._make_request(f\"content/{page_id}\", params=params)",
      "        if not data: return None",
      "        html_content = data.get(\"body\", {}).get(\"storage\", {}).get(\"value\", \"\")",
      "        clean_text = clean_html(html_content)",
      "        page_title = data.get(\"title\", \"Unknown Title\")",
      "        space_key = data.get(\"space\", {}).get(\"key\", \"UNK\")",
      "        version = data.get(\"version\", {}).get(\"number\", 0)",
      "        last_modified = data.get(\"version\", {}).get(\"when\", \"\")",
      "        page_url = f\"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}\"",
      "        if not clean_text: logger.warning(f\"Page ID {page_id} ('{page_title}') had no processable content.\")",
      "        return {\"id\": page_id, \"title\": page_title, \"url\": page_url, \"space_key\": space_key, \"version\": version, \"last_modified\": last_modified, \"text\": clean_text}",
      "",
      "",
      "# =========================================",
      "# === Ingestion Logic ===",
      "# =========================================",
      "def run_ingestion():",
      "    # Needs access to confluence_client and vector_db_collection",
      "    global confluence_client, vector_db_collection",
      "    if not confluence_client or not vector_db_collection:",
      "        logger.error(\"Clients not initialized. Cannot run ingestion.\")",
      "        return {\"status\": \"error\", \"message\": \"Clients not initialized\"}",
      "",
      "    logger.info(\"Starting Confluence data ingestion pipeline...\")",
      "    start_time = time.time()",
      "    pages = confluence_client.search_pages(query_terms=INGESTION_QUERY_TERMS, space_key=CONFLUENCE_SPACE_KEY, limit_per_req=50, max_total=500) # Limit total pages",
      "",
      "    if not pages:",
      "        logger.warning(\"No pages found matching the search criteria.\")",
      "        return {\"status\": \"complete\", \"message\": \"No relevant pages found\", \"pages_processed\": 0, \"chunks_added\": 0}",
      "",
      "    logger.info(f\"Found {len(pages)} potentially relevant pages.\")",
      "    all_chunks, all_metadatas, all_ids = [], [], []",
      "    processed_page_count = 0",
      "",
      "    for page_summary in pages:",
      "        page_id = page_summary.get('id') or page_summary.get('content', {}).get('id')",
      "        if not page_id: continue",
      "        page_details = confluence_client.get_page_details(page_id)",
      "        if not page_details or not page_details.get('text'): continue",
      "",
      "        logger.info(f\"Processing page: '{page_details['title']}' (ID: {page_id})\")",
      "        text = page_details['text']",
      "        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)",
      "        if not chunks: continue",
      "",
      "        for i, chunk in enumerate(chunks):",
      "            chunk_id = f\"{page_id}_chunk_{i}\"",
      "            metadata = {k: v for k, v in page_details.items() if k != 'text'} # Copy metadata, exclude text",
      "            metadata[\"chunk_index\"] = i",
      "            all_chunks.append(chunk)",
      "            all_metadatas.append(metadata)",
      "            all_ids.append(chunk_id)",
      "        processed_page_count += 1",
      "",
      "    chunks_added_count = 0",
      "    if all_chunks:",
      "        logger.info(f\"Preparing to upsert {len(all_chunks)} total chunks from {processed_page_count} pages.\")",
      "        try:",
      "            # Assuming ChromaDB client handles embedding via collection's function",
      "            vector_db_collection.upsert(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)",
      "            chunks_added_count = len(all_chunks)",
      "            logger.info(\"Upsert operation completed.\")",
      "            logger.info(f\"Vector DB now contains approximately {vector_db_collection.count()} documents.\")",
      "        except Exception as e:",
      "             logger.error(f\"Failed during vector DB upsert: {e}\", exc_info=True)",
      "             return {\"status\": \"error\", \"message\": f\"Vector DB upsert failed: {e}\", \"pages_processed\": processed_page_count, \"chunks_added\": 0}",
      "    else:",
      "        logger.warning(\"No valid chunks were generated to upsert.\")",
      "",
      "    end_time = time.time()",
      "    duration = end_time - start_time",
      "    logger.info(f\"Ingestion pipeline finished in {duration:.2f} seconds.\")",
      "    return {\"status\": \"complete\", \"message\": \"Ingestion finished successfully.\", \"pages_processed\": processed_page_count, \"chunks_added\": chunks_added_count, \"duration_seconds\": duration}",
      "",
      "# =========================================",
      "# === RAG Query Logic ===",
      "# =========================================",
      "async def get_rag_answer(question: str) -> str:",
      "    # Needs access to vector_db_collection, embedding_client_vertex, llm_client_gemini",
      "    global vector_db_collection, embedding_client_vertex, llm_client_gemini",
      "    if not all([vector_db_collection, embedding_client_vertex, llm_client_gemini]):",
      "        logger.error(\"Clients not initialized. Cannot run RAG query.\")",
      "        return \"Error: Backend services not ready.\"",
      "",
      "    logger.info(f\"Processing RAG query: '{question[:50]}...'\" )",
      "    try:",
      "        # 1. Embed the question",
      "        # Note: Vertex AI embedding model might prefer a list of texts",
      "        question_embedding_response = embedding_client_vertex.get_embeddings([question])",
      "        if not question_embedding_response or len(question_embedding_response) == 0:",
      "             raise ValueError(\"Failed to get embedding for the question.\")",
      "        question_embedding = question_embedding_response[0].values",
      "",
      "        # 2. Query Vector DB",
      "        search_results = vector_db_collection.query(",
      "            query_embeddings=[question_embedding],",
      "            n_results=CONTEXT_NUM_RESULTS,",
      "            include=['metadatas', 'documents'] # Only need these",
      "        )",
      "",
      "        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:",
      "            logger.warning(\"No relevant context found in Vector DB for the question.\")",
      "            context_str = \"No relevant context found in the knowledge base.\"",
      "        else:",
      "            context_docs = search_results['documents'][0]",
      "            metadatas = search_results['metadatas'][0]",
      "            context_parts = []",
      "            for i, doc in enumerate(context_docs):",
      "                meta = metadatas[i]",
      "                title = meta.get('title', 'N/A')",
      "                url = meta.get('url', 'N/A')",
      "                context_parts.append(f\"Source {i+1} (Title: {title}, URL: {url}):\\n{doc}\")",
      "            context_str = \"\\n\\n---\\n\\n\".join(context_parts)",
      "            logger.info(f\"Retrieved {len(context_docs)} context documents.\")",
      "",
      "        # 3. Construct Prompt & Call LLM",
      "        system_instruction = (",
      "            \"You are a helpful assistant knowledgeable about Copper APIs based on internal Confluence documentation. \"",
      "            \"Answer the user's question using *only* the provided context from Confluence pages below. \"",
      "            \"If the context doesn't contain the answer, clearly state that. Cite sources if possible.\"",
      "        )",
      "        MAX_CONTEXT_CHARS = 28000 # Truncate if needed",
      "        if len(context_str) > MAX_CONTEXT_CHARS: context_str = context_str[:MAX_CONTEXT_CHARS] + \"...[Truncated]\"",
      "        prompt_parts = [",
      "            Part.from_text(f\"Context:\\n{context_str}\"),",
      "            Part.from_text(f\"\\nQuestion: {question}\\n\\nAnswer:\")",
      "        ]",
      "",
      "        # Using the global llm_client_gemini",
      "        llm_response = llm_client_gemini.generate_content(",
      "             prompt_parts,",
      "             generation_config=GenerationConfig(temperature=0.3, max_output_tokens=4096),",
      "             safety_settings={k: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE for k in HarmCategory},",
      "             stream=False, # Keep simple for now",
      "             system_instruction=system_instruction",
      "         )",
      "",
      "        # 4. Process LLM Response",
      "        if llm_response.candidates and llm_response.candidates[0].content.parts:",
      "            answer = llm_response.candidates[0].content.parts[0].text",
      "            logger.info(f\"RAG query successful. Answer length: {len(answer)}\")",
      "            return answer",
      "        else:",
      "             finish_reason = llm_response.candidates[0].finish_reason if llm_response.candidates else 'N/A'",
      "             logger.warning(f\"LLM response was empty or blocked. Finish Reason: {finish_reason}\")",
      "             return \"Error: The language model did not return a valid response. It may have been blocked.\"",
      "",
      "    except Exception as e:",
      "        logger.error(f\"Error during RAG processing for question '{question[:50]}...': {e}\", exc_info=True)",
      "        return f\"Error: An internal error occurred while processing your request: {e}\"",
      "",
      "# =========================================",
      "# === FastAPI Application ===",
      "# =========================================",
      "",
      "# --- Pydantic Models ---",
      "class QueryRequest(BaseModel):",
      "    question: str",
      "",
      "class QueryResponse(BaseModel):",
      "    answer: str",
      "",
      "class IngestionResponse(BaseModel):",
      "    status: str",
      "    message: str",
      "    pages_processed: int | None = None",
      "    chunks_added: int | None = None",
      "    duration_seconds: float | None = None",
      "",
      "# --- Lifespan for Client Initialization ---",
      "@asynccontextmanager",
      "async def lifespan(app: FastAPI):",
      "    global confluence_client, vector_db_collection, llm_client_gemini, embedding_client_vertex",
      "    logger.info(\"API starting up - Initializing clients...\")",
      "    try:",
      "        # Init Confluence Client",
      "        confluence_client = _ConfluenceClient()",
      "        logger.info(\"Confluence client initialized.\")",
      "",
      "        # Init Vertex AI (common step)",
      "        vertexai.init(project=PROJECT_ID, location=REGION)",
      "        logger.info(\"Vertex AI initialized.\")",
      "",
      "        # Init Gemini LLM Client",
      "        llm_client_gemini = GenerativeModel(GEMINI_MODEL_NAME)",
      "        logger.info(\"Gemini LLM client initialized.\")",
      "",
      "        # Init Vertex AI Embedding Client",
      "        embedding_client_vertex = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)",
      "        logger.info(\"Vertex AI Embedding client initialized.\")",
      "",
      "        # Init ChromaDB Client and Collection",
      "        # Note: Vertex AI Embeddings via google-cloud-aiplatform might need separate function or library setup",
      "        # Using a ChromaDB-provided generic one as fallback if direct Vertex fails",
      "        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)",
      "        # Attempt direct Vertex function (ensure library compatibility)",
      "        # try:",
      "        #    chroma_embedding_function = embedding_functions.GoogleVertexEmbeddingFunction(",
      "        #         project_id=PROJECT_ID,",
      "        #         location=REGION,",
      "        #         model_name=EMBEDDING_MODEL_NAME",
      "        #    )",
      "        # except ImportError: # Fallback if library not installed or incompatible",
      "        logger.warning(\"Direct GoogleVertexEmbeddingFunction for Chroma might require specific setup. Using SentenceTransformer as fallback.\")",
      "        chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=\"all-MiniLM-L6-v2\")",
      "",
      "        vector_db_collection = chroma_client.get_or_create_collection(",
      "            name=COLLECTION_NAME,",
      "            embedding_function=chroma_embedding_function,",
      "            metadata={\"hnsw:space\": \"cosine\"}",
      "        )",
      "        logger.info(f\"ChromaDB collection '{COLLECTION_NAME}' loaded/created. Count: {vector_db_collection.count()}\")",
      "",
      "        logger.info(\"All clients initialized successfully.\")",
      "    except Exception as e:",
      "        logger.critical(f\"FATAL: Failed to initialize resources during startup: {e}\", exc_info=True)",
      "        # Stop the application from starting if critical components fail",
      "        raise RuntimeError(\"Critical resource initialization failed.\") from e",
      "    yield",
      "    # Cleanup (if necessary)",
      "    logger.info(\"API shutting down...\")",
      "",
      "app = FastAPI(title=\"Unified Confluence RAG Backend\", lifespan=lifespan)",
      "",
      "# --- API Endpoints ---",
      "@app.get(\"/health\")",
      "async def health_check():",
      "    # Add more checks if needed (e.g., vector db count)",
      "    return {\"status\": \"ok\", \"vector_db_collection\": COLLECTION_NAME, \"doc_count\": vector_db_collection.count() if vector_db_collection else 'N/A'}",
      "",
      "@app.post(\"/ingest\", response_model=IngestionResponse)",
      "async def trigger_ingestion():",
      "    \"\"\"Triggers the Confluence ingestion process. Runs synchronously for simplicity.\"",
      "    logger.info(\"Received request to trigger ingestion.\")",
      "    # WARNING: Running ingestion synchronously will block the API.",
      "    # In production, use background tasks (Celery, K8s Jobs, Cloud Run Jobs, etc.)",
      "    try:",
      "        result = run_ingestion() # Call the synchronous function",
      "        if result[\"status\"] == \"error\":",
      "             raise HTTPException(status_code=500, detail=result[\"message\"])",
      "        return IngestionResponse(**result)",
      "    except Exception as e:",
      "        logger.error(f\"Ingestion endpoint failed unexpectedly: {e}\", exc_info=True)",
      "        raise HTTPException(status_code=500, detail=f\"Internal server error during ingestion: {e}\")",
      "",
      "@app.post(\"/query\", response_model=QueryResponse)",
      "async def ask_question(query: QueryRequest):",
      "    \"\"\"Receives a question and returns an answer generated via RAG.\"\"\"",
      "    logger.info(f\"Received query request: '{query.question[:50]}...'\" )",
      "    try:",
      "        answer = await get_rag_answer(query.question) # Call the async RAG function",
      "        if answer.startswith(\"Error:\"):",
      "             # Determine if it's a 'not found' or a 'processing' error",
      "             if \"not found\" in answer or \"not ready\" in answer:",
      "                 raise HTTPException(status_code=404, detail=answer)",
      "             else:",
      "                 raise HTTPException(status_code=500, detail=answer)",
      "        return QueryResponse(answer=answer)",
      "    except HTTPException as http_exc:",
      "         raise http_exc # Re-raise specific HTTP errors",
      "    except Exception as e:",
      "        logger.error(f\"Error processing query '{query.question[:50]}...': {e}\", exc_info=True)",
      "        raise HTTPException(status_code=500, detail=f\"Internal server error: {e}\")",
      "",
      "",
      "# =========================================",
      "# === Main Execution ===",
      "# =========================================",
      "if __name__ == \"__main__\":",
      "    logger.info(f\"Starting Unified Backend API server on {API_HOST}:{API_PORT}\")",
      "    # Run uvicorn programmatically",
      "    # Note: reload=True is useful for development but should be False in production",
      "    uvicorn.run(app, host=API_HOST, port=API_PORT)",
      ""
    ],
    "required_external_files": [
      "`.env` (with your credentials and configuration)",
      "`requirements.txt` (to install dependencies)"
    ],
    "how_to_run": [
      "Save the code as `unified_backend.py`.",
      "Create a `.env` file in the same directory with your credentials.",
      "Create a virtual environment: `python -m venv venv`",
      "Activate it: `source venv/bin/activate` (or `venv\\Scripts\\activate` on Windows)",
      "Install requirements: `pip install -r requirements.txt` (You'll need to create requirements.txt with the listed dependencies)",
      "Ensure Google Cloud authentication is set up (`gcloud auth application-default login`).",
      "Run the script: `python unified_backend.py`",
      "The API will start on `http://0.0.0.0:8000` (or your configured host/port).",
      "**Important First Step:** Trigger ingestion by sending a POST request to `http://localhost:8000/ingest` (e.g., using `curl -X POST http://localhost:8000/ingest` or Postman). Monitor the console logs.",
      "Once ingestion is complete, send questions via POST requests to `http://localhost:8000/query` with a JSON body like `{\"question\": \"Your question here\"}`.",
      "Check the health endpoint: `http://localhost:8000/health`."
    ]
  },
  "final_warning": "This single-file approach is **strongly discouraged** for anything beyond initial prototyping or a hackathon demonstration where refactoring is planned. It tightly couples different concerns, making it hard to test, debug, scale, and maintain. Please break this down into logical modules (clients, core logic, API, ingestion) for any further development."
}
