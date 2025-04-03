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
