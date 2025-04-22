#!/usr/bin/env python3
"""
BMC Helix + Gemini Chatbot Integration (Final Working Version)

This application integrates BMC Helix with Gemini AI to create a professional chatbot
that answers questions about incidents, service requests, and other BMC Helix data.

Uses the exact same Gemini API method that works in your office environment.

Usage:
    python3 final_working_helix_gemini_chatbot.py
"""

import os
import sys
import logging
import json
import re
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Gemini imports - using the exact structure from your working example
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("helix_gemini_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HelixGeminiChatbot")

# Configuration (Hardcoded as per requirements)
PROJECT_ID = "prj-dv-cws-4363"
REGION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"

# BMC Helix configuration - replace these with actual credentials
BMC_SERVER = "cmegroup-restapi.onbmc.com"
BMC_USERNAME = "username"  # Replace with actual username
BMC_PASSWORD = "password"  # Replace with actual password

class BMCHelixAPI:
    """Client for interacting with the BMC Helix API."""
    
    def __init__(self, server=BMC_SERVER, username=BMC_USERNAME, password=BMC_PASSWORD):
        """Initialize the BMC Helix API client."""
        self.server = server
        self.username = username
        self.password = password
        self.token = None
        self.headers = None
    
    def login(self) -> bool:
        """Login to BMC Helix and get an authentication token."""
        logger.info(f"Logging into BMC Helix at {self.server}")
        url = f"https://{self.server}/api/jwt/login"
        payload = {'username': self.username, 'password': self.password}
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=False)
            
            if r.status_code == 200:
                self.token = r.text
                self.headers = {'Authorization': f'AR-JWT {self.token}'}
                logger.info("Login successful")
                return True
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            return False
    
    def logout(self) -> bool:
        """Logout from BMC Helix."""
        if not self.token:
            logger.warning("Not logged in, cannot logout")
            return True
            
        url = f"https://{self.server}/api/jwt/logout"
        
        try:
            r = requests.post(url, headers=self.headers, verify=False)
            
            if r.status_code == 204:
                logger.info("Logout successful")
                self.token = None
                self.headers = None
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            return False
    
    def get_incidents(self, query_params=None) -> List[Dict]:
        """Get incidents from BMC Helix."""
        if not self.token:
            logger.warning("Not logged in, attempting to login")
            if not self.login():
                return []
        
        # Specify only the most important fields to avoid errors
        fields = "Status,Summary,Support Group Name,Request Assignee,Submitter,Incident Number,Submit Date,Priority"
        
        url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help%20Desk"
        
        # Build params properly
        params = {}
        
        # Add the fields parameter
        params['fields'] = fields
        
        # Add limit parameter
        params['limit'] = 30
        
        # Add query parameter if provided
        if query_params and 'q' in query_params:
            params['q'] = query_params['q']
        
        logger.info(f"Fetching incidents with params: {params}")
        
        try:
            r = requests.get(url, headers=self.headers, params=params, verify=False)
            
            if r.status_code == 200:
                data = r.json()
                entries = data.get('entries', [])
                logger.info(f"Retrieved {len(entries)} incidents")
                
                # Normalize the data structure
                incidents = []
                for entry in entries:
                    incident = {}
                    for key, value in entry.get('values', {}).items():
                        incident[key] = value
                    incidents.append(incident)
                return incidents
            elif r.status_code == 401:
                logger.warning("Unauthorized access. Attempting to re-login.")
                if self.login():
                    # Try again with new token
                    return self.get_incidents(query_params)
                return []
            else:
                logger.error(f"Failed to get incidents with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting incidents: {str(e)}")
            return []
    
    def get_incident_by_id(self, incident_id: str) -> Optional[Dict]:
        """Get a specific incident by ID."""
        # Sanitize the incident ID
        incident_id = incident_id.replace('"', '').replace("'", "").strip()
        
        query_params = {
            'q': f"'Incident Number'=\"{incident_id}\""
        }
        
        incidents = self.get_incidents(query_params)
        
        if incidents and len(incidents) > 0:
            return incidents[0]
        else:
            logger.warning(f"No incident found with ID: {incident_id}")
            return None
    
    def get_incidents_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get incidents within a date range."""
        # Format dates for BMC Helix query
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        query_params = {
            'q': f"'Submit Date' >= \"{start_date_str}\" AND 'Submit Date' <= \"{end_date_str}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_status(self, status: str) -> List[Dict]:
        """Get incidents with a specific status."""
        query_params = {
            'q': f"'Status'=\"{status}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_priority(self, priority: str) -> List[Dict]:
        """Get incidents with a specific priority."""
        query_params = {
            'q': f"'Priority'=\"{priority}\""
        }
        
        return self.get_incidents(query_params)
    
    def search_incidents(self, search_term: str) -> List[Dict]:
        """Search for incidents containing a specific term in summary."""
        # Sanitize the search term
        search_term = search_term.replace('"', '\\"')
        
        query_params = {
            'q': f"'Summary' LIKE \"%{search_term}%\""
        }
        
        return self.get_incidents(query_params)


class GeminiClient:
    """Client for interacting with Google's Gemini AI model using Vertex AI."""
    
    def __init__(self, project_id=PROJECT_ID, location=REGION, model_name=MODEL_NAME):
        """Initialize the Gemini client."""
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.model = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Vertex AI and create Gemini model instance."""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Create model instance
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Using model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def generate_response(self, prompt: str, system_instructions: Optional[str] = None) -> str:
        """
        Generate a response from the Gemini model.
        
        Args:
            prompt (str): The prompt to send to the model
            system_instructions (str, optional): System instructions to guide the model
            
        Returns:
            str: The generated response
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return "Error: Unable to connect to Gemini AI"
        
        # Combine system instructions with prompt if provided
        if system_instructions:
            full_prompt = f"{system_instructions}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        logger.info(f"Generating response for prompt: {full_prompt[:100]}...")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.2,  # Lower temperature for more focused responses
                top_p=0.95,
                max_output_tokens=4096,
            )
            
            # Generate response (non-streaming for simplicity)
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                response_text = response.text
            else:
                # Handle various response structures
                response_text = ""
                if hasattr(response, 'candidates') and response.candidates:
                    if hasattr(response.candidates[0], 'text'):
                        response_text = response.candidates[0].text
                    elif hasattr(response.candidates[0], 'content'):
                        if hasattr(response.candidates[0].content, 'text'):
                            response_text = response.candidates[0].content.text
                        elif hasattr(response.candidates[0].content, 'parts'):
                            parts = response.candidates[0].content.parts
                            if parts and hasattr(parts[0], 'text'):
                                response_text = parts[0].text
            
            logger.info(f"Generated response of length {len(response_text)}")
            return response_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class QueryProcessor:
    """Process and interpret user queries."""
    
    def __init__(self, helix_api: BMCHelixAPI, gemini_client: GeminiClient):
        """Initialize the query processor."""
        self.helix_api = helix_api
        self.gemini_client = gemini_client
        
        # Define system instructions
        self.system_instructions = """
        You are a professional and friendly assistant that helps answer questions about BMC Helix incidents 
        and service requests. Your responses should be clear, concise, and well-structured.
        """
        
        # Instructions for analyzing incident data
        self.incident_analysis_instructions = """
        You are a professional IT service management assistant analyzing BMC Helix incident data.
        
        The incident data includes fields such as Incident Number, Status, Priority, Summary, etc.
        
        When analyzing incidents:
        - Group by relevant categories (status, priority, assigned group)
        - Identify important patterns
        - Highlight critical items
        - Summarize the overall status
        
        Format your response with appropriate tables and clearly organized sections.
        """
    
    def process_query(self, query: str) -> str:
        """Process a user query and generate a response."""
        logger.info(f"Processing query: {query}")
        
        try:
            # First, determine what the user is asking about
            if "incident" in query.lower() and any(term in query.lower() for term in ["yesterday", "last week", "recent"]):
                # User is asking about recent incidents
                end_date = datetime.now()
                
                if "yesterday" in query.lower():
                    start_date = end_date - timedelta(days=1)
                    incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
                elif "last week" in query.lower():
                    start_date = end_date - timedelta(days=7)
                    incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
                else:
                    # Default to last 3 days
                    start_date = end_date - timedelta(days=3)
                    incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
                
                # Filter by priority if specified
                if "high priority" in query.lower():
                    incidents = [inc for inc in incidents if inc.get("Priority", "").lower() == "high"]
                elif "critical" in query.lower():
                    incidents = [inc for inc in incidents if inc.get("Priority", "").lower() == "critical"]
                
                # Convert incidents to JSON for analysis
                incidents_json = json.dumps(incidents, indent=2)
                
                # Create prompt for Gemini
                analysis_prompt = f"""
                The user asked: "{query}"
                
                Here are the incidents from the specified time period:
                ```json
                {incidents_json}
                ```
                
                Please analyze these incidents and provide a helpful response that addresses the query.
                Include counts by status and priority, and highlight any important patterns.
                Format your response with appropriate tables and sections.
                """
                
                # Get analysis from Gemini
                response = self.gemini_client.generate_response(analysis_prompt, self.incident_analysis_instructions)
                return response
                
            elif "incident" in query.lower() and "INC" in query:
                # User is asking about a specific incident
                # Extract the incident ID
                incident_id = None
                match = re.search(r'INC\d+', query)
                if match:
                    incident_id = match.group(0)
                
                if incident_id:
                    incident = self.helix_api.get_incident_by_id(incident_id)
                    
                    if incident:
                        # Convert incident to JSON for analysis
                        incident_json = json.dumps(incident, indent=2)
                        
                        # Create prompt for Gemini
                        analysis_prompt = f"""
                        The user asked: "{query}"
                        
                        Here is the incident data:
                        ```json
                        {incident_json}
                        ```
                        
                        Please provide a detailed summary of this incident, including its current status, 
                        priority, and other relevant details. Format your response in a clear, professional manner.
                        """
                        
                        # Get analysis from Gemini
                        response = self.gemini_client.generate_response(analysis_prompt, self.incident_analysis_instructions)
                        return response
                    else:
                        return f"I couldn't find any incident with ID {incident_id}. Please check the incident number and try again."
                else:
                    return "I couldn't identify a specific incident ID in your query. Please include the incident ID (e.g., INC12345) in your query."
            
            elif "trend" in query.lower() or "pattern" in query.lower():
                # User is asking about trends or patterns
                # Get incidents from the last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
                
                # Convert incidents to JSON for analysis
                incidents_json = json.dumps(incidents, indent=2)
                
                # Create prompt for Gemini
                analysis_prompt = f"""
                The user asked: "{query}"
                
                Here are the incidents from the last 30 days:
                ```json
                {incidents_json}
                ```
                
                Please analyze these incidents for trends and patterns. Consider incident volume, 
                priority distribution, and common themes in incident summaries.
                Present your findings in a clear, structured format with appropriate tables and sections.
                """
                
                # Get analysis from Gemini
                response = self.gemini_client.generate_response(analysis_prompt, self.incident_analysis_instructions)
                return response
            
            else:
                # For other queries, get recent incidents as context
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
                
                # Convert incidents to JSON for analysis
                incidents_json = json.dumps(incidents, indent=2)
                
                # Create prompt for Gemini
                analysis_prompt = f"""
                The user asked: "{query}"
                
                Here are the incidents from the last 7 days:
                ```json
                {incidents_json}
                ```
                
                Please analyze these incidents and provide a response that best addresses the user's query.
                If the query doesn't relate to these incidents, please let the user know and suggest 
                how they might rephrase their question to get better results.
                """
                
                # Get analysis from Gemini
                response = self.gemini_client.generate_response(analysis_prompt, self.system_instructions)
                return response
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while processing your query: {error_msg}. Please try a different query or contact support if the issue persists."


class HelixGeminiChatbot:
    """Main class for the BMC Helix + Gemini Chatbot."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.helix_api = BMCHelixAPI()
        self.gemini_client = GeminiClient()
        self.query_processor = QueryProcessor(self.helix_api, self.gemini_client)
        
        # Try to login to BMC Helix
        login_successful = self.helix_api.login()
        if not login_successful:
            logger.error("Failed to login to BMC Helix. Chatbot may not function correctly.")
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a response."""
        return self.query_processor.process_query(query)
    
    def close(self):
        """Clean up resources and logout."""
        self.helix_api.logout()


def interactive_mode():
    """Run the chatbot in interactive mode."""
    print("="*50)
    print("BMC Helix + Gemini Chatbot")
    print("="*50)
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Ask questions about incidents, e.g.:")
    print("- What incidents were created yesterday?")
    print("- Show me all high priority incidents from last week")
    print("- Give me details about incident INC123456")
    print("-"*50)
    
    chatbot = HelixGeminiChatbot()
    
    try:
        while True:
            query = input("\nYour query: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            if not query:
                continue
            
            start_time = time.time()
            print("\nProcessing your query... (this may take a moment)")
            
            response = chatbot.process_query(query)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print("\n" + "="*80)
            print(response)
            print("="*80)
            print(f"\nResponse generated in {processing_time:.2f} seconds.")
    
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nClosing session...")
        chatbot.close()
        print("Session closed.")


def main():
    """Main entry point for the application."""
    interactive_mode()


if __name__ == "__main__":
    main()














#!/usr/bin/env python3
#########################################################
# BMC Helix + Gemini AI Incident Analysis Chatbot
# Features:
# - Integrated with BMC Helix REST API
# - Gemini AI for natural language processing
# - Automatic date range detection
# - Smart data categorization
# - Professional chat interface
# - Built-in error handling
#########################################################

import os
import json
import requests
import vertexai
from datetime import datetime, timedelta
from dateutil import parser
from vertexai.generative_models import GenerativeModel, GenerationConfig
import logging
import urllib3
import getpass

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration - Hardcoded for testing (Update these values)
CONFIG = {
    "BMC": {
        "server": "cmegroup-restapi.onbmc.com",
        "username": "YOUR_SERVICE_ACCOUNT",
        "password": "YOUR_PASSWORD",
        "incident_form": "HPD:IncidentInterface"
    },
    "GEMINI": {
        "project_id": "pri-dv-cws-4363",
        "location": "us-central1",
        "model_name": "gemini-2.0-flash-001"
    },
    "DATE_FORMAT": "%Y-%m-%dT%H:%M:%SZ"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("helix_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HelixChatbot")

class HelixConnector:
    def __init__(self):
        self.token = None
        self.session = requests.Session()
        self.session.verify = False
        
    def login(self):
        """Authenticate with BMC Helix"""
        url = f"https://{CONFIG['BMC']['server']}/api/jwt/login"
        try:
            response = self.session.post(
                url,
                data={'username': CONFIG['BMC']['username'], 
                      'password': CONFIG['BMC']['password']},
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            if response.status_code == 200:
                self.token = response.text
                logger.info("BMC Helix login successful")
                return True
            logger.error(f"Login failed: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Login exception: {str(e)}")
            return False

    def get_incidents(self, days_back=7):
        """Retrieve incidents from BMC Helix"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        query = (
            f"'Submit Date' >= \"{start_date.strftime(CONFIG['DATE_FORMAT'])}\""
            f" AND 'Submit Date' <= \"{end_date.strftime(CONFIG['DATE_FORMAT'])}\""
        )
        
        url = (
            f"https://{CONFIG['BMC']['server']}/api/arsys/v1/entry/{CONFIG['BMC']['incident_form']}"
            f"?q={query}"
            "&fields=Incident Number,Status,Priority,Submit Date,Assigned Group,Description,Category"
        )
        
        try:
            response = self.session.get(
                url,
                headers={'Authorization': f'AR-JWT {self.token}'}
            )
            if response.status_code == 200:
                return response.json().get('entries', [])
            logger.error(f"Incident fetch failed: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Incident fetch exception: {str(e)}")
            return None

class GeminiAnalyzer:
    def __init__(self):
        vertexai.init(
            project=CONFIG['GEMINI']['project_id'],
            location=CONFIG['GEMINI']['location']
        )
        self.model = GenerativeModel(CONFIG['GEMINI']['model_name'])
        self.config = GenerationConfig(
            temperature=0.3,
            top_p=0.9,
            max_output_tokens=8192
        )

    def analyze_incidents(self, incidents, query):
        """Analyze incidents using Gemini AI"""
        prompt = f"""
        You are a professional IT support analyst. Analyze these {len(incidents)} incidents 
        based on the user query: "{query}". 

        Instructions:
        1. Understand the user's intent and required analysis type
        2. Categorize incidents appropriately
        3. Create a professional summary
        4. Generate a markdown table with key details
        5. Highlight critical issues if any

        Incident Data:
        {json.dumps(incidents, indent=2)}

        Respond in this format:
        **Summary**: [concise analysis]
        
        **Recommendations**: [bullet points]
        
        **Incident Details**:
        | Column1 | Column2 | ... |
        |---------|---------|-----|
        | ...     | ...     | ... |
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            return "Sorry, I encountered an error processing your request."

class ChatbotEngine:
    def __init__(self):
        self.helix = HelixConnector()
        self.gemini = GeminiAnalyzer()
        
    def process_query(self, query):
        """Main processing pipeline"""
        # Detect date range in query
        days_back = self._detect_date_range(query)
        
        # Fetch incidents
        if not self.helix.login():
            return "Error: Failed to connect to BMC Helix"
            
        incidents = self.helix.get_incidents(days_back)
        if not incidents:
            return "No incidents found for the specified period."
            
        # Process with Gemini
        raw_incidents = [entry['values'] for entry in incidents]
        return self.gemini.analyze_incidents(raw_incidents, query)

    def _detect_date_range(self, text):
        """Auto-detect date range from natural language"""
        text = text.lower()
        if "yesterday" in text:
            return 1
        if "week" in text:
            return 7
        if "month" in text:
            return 30
        return 7  # Default to 1 week

def main():
    print("\n=== BMC Helix AI Assistant ===")
    print("Type 'exit' to quit\n")
    
    chatbot = ChatbotEngine()
    
    while True:
        try:
            query = input("\nHow can I assist you today?\n> ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                break
                
            print("\nAnalyzing your request...\n")
            result = chatbot.process_query(query)
            print("\n" + "-"*50)
            print(result)
            print("-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error(f"Main loop error: {str(e)}")

if __name__ == "__main__":
    main()