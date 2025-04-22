#!/usr/bin/env python3
"""
BMC Helix + Gemini Chatbot Integration (Fixed Version)

This application integrates BMC Helix with Gemini AI to create a professional chatbot
that answers questions about incidents, service requests, and other BMC Helix data.

This is the fixed version addressing API compatibility issues and error handling.

Usage:
    python3 fixed_helix_gemini_chatbot.py
"""

import os
import sys
import logging
import json
import re
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

# Google Gemini imports - using proper import path
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

# Disable SSL warnings (for development only, consider proper cert handling in production)
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

# Configuration (Hardcoded for testing as per requirements)
PROJECT_ID = "prj-dv-cws-4363"
REGION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"
API_KEY = None  # Set this if using API key authentication instead of service account

# BMC Helix configuration
BMC_SERVER = "cmegroup-restapi.onbmc.com"
BMC_USERNAME = "username"  # Replace with actual username for production
BMC_PASSWORD = "password"  # Replace with actual password for production

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
        """
        Login to BMC Helix and get an authentication token.
        
        Returns:
            bool: True if login was successful, False otherwise
        """
        logger.info(f"Logging into BMC Helix at {self.server}")
        url = f"https://{self.server}/api/jwt/login"
        payload = {'username': self.username, 'password': self.password}
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=False)
            
            if r.status_code == 200:
                self.token = r.text
                self.headers = {'Authorization': f'AR-JWT {self.token}', 'Content-Type': 'application/json'}
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
        """
        Logout from BMC Helix.
        
        Returns:
            bool: True if logout was successful, False otherwise
        """
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
    
    def _check_token_and_retry(self, func, *args, **kwargs):
        """
        Check if token is valid, retry with new token if needed.
        
        Args:
            func: Function to retry
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Result of function call or empty result on failure
        """
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.warning("Token may have expired, attempting to re-login")
                if self.login():
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as retry_e:
                        logger.error(f"Error after token refresh: {str(retry_e)}")
                        return []
                else:
                    logger.error("Failed to refresh token")
                    return []
            else:
                logger.error(f"Error calling function: {str(e)}")
                return []
    
    def get_incidents(self, query_params=None) -> List[Dict]:
        """
        Get incidents from BMC Helix.
        
        Args:
            query_params (Dict): Query parameters to filter incidents
            
        Returns:
            List[Dict]: List of incidents
        """
        if not self.token:
            logger.warning("Not logged in, attempting to login")
            if not self.login():
                return []
        
        # Define fields to retrieve - explicitly listing them to avoid errors
        fields = "Status,Summary,Support Group Name,Request Assignee,Submitter,Work Order ID,Request Manager,Incident Number,Description,Status,Owner,Impact,Owner Group,Submit Date,Assigned Group,Priority,Environment"
        
        url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help%20Desk"
        
        params = {
            'fields': fields,
            'limit': 50  # Limiting to 50 records to avoid performance issues
        }
        
        if query_params:
            # Ensure query parameter is properly formatted
            if 'q' in query_params:
                params['q'] = query_params['q']
            # Add any other parameters
            for key, value in query_params.items():
                if key != 'q' and key != 'fields' and key != 'limit':
                    params[key] = value
        
        try:
            logger.info(f"Fetching incidents with params: {params}")
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
                    return self._check_token_and_retry(self.get_incidents, query_params)
                return []
            else:
                logger.error(f"Failed to get incidents with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting incidents: {str(e)}")
            return []
    
    def get_incident_by_id(self, incident_id: str) -> Optional[Dict]:
        """
        Get a specific incident by ID.
        
        Args:
            incident_id (str): The incident ID to fetch
            
        Returns:
            Optional[Dict]: The incident data if found, None otherwise
        """
        # Remove quotes to handle potential input format issues
        incident_id = incident_id.replace('"', '').replace("'", "")
        
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
        """
        Get incidents within a date range.
        
        Args:
            start_date (datetime): Start date for incident search
            end_date (datetime): End date for incident search
            
        Returns:
            List[Dict]: List of incidents in the date range
        """
        # Format dates for BMC Helix query
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        query_params = {
            'q': f"'Submit Date' >= \"{start_date_str}\" AND 'Submit Date' <= \"{end_date_str}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_status(self, status: str) -> List[Dict]:
        """
        Get incidents with a specific status.
        
        Args:
            status (str): Status to filter by (e.g., "Open", "Closed", "Resolved")
            
        Returns:
            List[Dict]: List of incidents with the specified status
        """
        query_params = {
            'q': f"'Status'=\"{status}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_support_group(self, support_group: str) -> List[Dict]:
        """
        Get incidents assigned to a specific support group.
        
        Args:
            support_group (str): Support group to filter by
            
        Returns:
            List[Dict]: List of incidents assigned to the specified support group
        """
        query_params = {
            'q': f"'Support Group Name'=\"{support_group}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_priority(self, priority: str) -> List[Dict]:
        """
        Get incidents with a specific priority.
        
        Args:
            priority (str): Priority to filter by (e.g., "Critical", "High", "Medium", "Low")
            
        Returns:
            List[Dict]: List of incidents with the specified priority
        """
        query_params = {
            'q': f"'Priority'=\"{priority}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_assignee(self, assignee: str) -> List[Dict]:
        """
        Get incidents assigned to a specific user.
        
        Args:
            assignee (str): Assignee to filter by
            
        Returns:
            List[Dict]: List of incidents assigned to the specified user
        """
        query_params = {
            'q': f"'Request Assignee'=\"{assignee}\""
        }
        
        return self.get_incidents(query_params)
    
    def search_incidents(self, search_term: str) -> List[Dict]:
        """
        Search for incidents containing a specific term in summary or description.
        
        Args:
            search_term (str): Term to search for
            
        Returns:
            List[Dict]: List of incidents matching the search term
        """
        # Escape quotes in search term
        search_term = search_term.replace('"', '\\"')
        
        query_params = {
            'q': f"'Summary' LIKE \"%{search_term}%\" OR 'Description' LIKE \"%{search_term}%\""
        }
        
        return self.get_incidents(query_params)
    
    def execute_custom_query(self, query: str) -> List[Dict]:
        """
        Execute a custom query against the BMC Helix API.
        
        Args:
            query (str): Custom query string
            
        Returns:
            List[Dict]: Results of the custom query
        """
        query_params = {
            'q': query
        }
        
        return self.get_incidents(query_params)


class GeminiClient:
    """Client for interacting with Google's Gemini AI model."""
    
    def __init__(self, project_id=PROJECT_ID, region=REGION, model_name=MODEL_NAME, api_key=API_KEY):
        """Initialize the Gemini client."""
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.api_key = api_key
        self.model = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize the Gemini client."""
        try:
            # Configure with API key if provided, otherwise use application default credentials
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                genai.configure(project_id=self.project_id, location=self.region)
                
            # Initialize the model
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def generate_response(self, system_instructions: str, prompt: str) -> str:
        """
        Generate a response from the Gemini model.
        
        Args:
            system_instructions (str): System instructions to guide the model
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The generated response
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return "Error: Unable to connect to Gemini AI"
        
        logger.info(f"Generating response for prompt: {prompt[:100]}...")
        
        try:
            # Combine system instructions and prompt
            full_prompt = prompt
            if system_instructions:
                full_prompt = f"{system_instructions}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more deterministic responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            # Generate the response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            logger.info(f"Response generated with {len(response.text)} characters")
            return response.text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class QueryProcessor:
    """Process and interpret user queries."""
    
    def __init__(self, helix_api: BMCHelixAPI, gemini_client: GeminiClient):
        """
        Initialize the query processor.
        
        Args:
            helix_api (BMCHelixAPI): Instance of BMC Helix API client
            gemini_client (GeminiClient): Instance of Gemini client
        """
        self.helix_api = helix_api
        self.gemini_client = gemini_client
        
        # Define system instructions for different types of queries
        self.system_instructions = """
        You are a professional and friendly assistant that helps answer questions about BMC Helix incidents and service requests.
        You should respond in a clear, concise, and structured manner that is appropriate for a professional IT service management environment.
        
        When relevant, you should format your responses with tables, bullet points, or other structures to enhance readability.
        
        You will be given data from BMC Helix, and your task is to analyze and present this data in a way that addresses the user's query effectively.
        
        Your responses should be:
        1. Professional and courteous
        2. Clear and concise
        3. Structured for easy reading
        4. Accurate based on the data provided
        
        If the data is empty or you cannot answer the question based on the provided data, politely explain this and suggest alternative approaches.
        """
        
        # Instructions specifically for analyzing incident data
        self.incident_analysis_instructions = """
        You are a professional and friendly IT service management assistant. Your task is to analyze BMC Helix incident data and generate insights based on the user's query.
        
        The incident data may include fields such as:
        - Incident Number
        - Status
        - Priority
        - Summary
        - Description
        - Assigned Group
        - Assignee
        - Submit Date
        - Requester
        - and others
        
        When analyzing incidents, consider:
        - Grouping by relevant categories (status, priority, assigned group, etc.)
        - Identifying patterns or trends
        - Highlighting critical or high-priority items
        - Summarizing the overall status
        
        Format your response with appropriate tables, summaries, and insights. Always provide a brief introduction explaining what data you analyzed and a conclusion with key takeaways.
        
        If asked to compare or trend data over time, use clear descriptions of any patterns or changes observed.
        
        Present your information using well-formatted tables and properly organized text. Use spacing and formatting to make the response easy to read.
        Be thorough and make sure your analysis is comprehensive, capturing all the important aspects of the data.
        """
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The response to the query
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # First, let's use Gemini to interpret what kind of query this is
            interpretation_prompt = f"""
            Analyze the following query about BMC Helix incidents to determine what information is being requested:
            
            Query: "{query}"
            
            Provide your analysis in JSON format with these fields:
            {{
                "query_type": "incident_list" | "incident_details" | "incident_statistics" | "incident_search" | "other",
                "time_period": "specific_date" | "yesterday" | "last_week" | "last_month" | "date_range" | "not_specified",
                "specific_date": null,
                "date_range_start": null,
                "date_range_end": null,
                "status_filter": "all" | "open" | "closed" | "resolved" | "in_progress" | "not_specified",
                "priority_filter": "all" | "critical" | "high" | "medium" | "low" | "not_specified",
                "group_filter": null,
                "assignee_filter": null,
                "search_terms": null,
                "requested_analysis": "categorization" | "summary" | "details" | "trends" | "not_specified",
                "specific_incident_id": null
            }}
            
            Return ONLY the JSON object without any additional text.
            """
            
            # Get query interpretation from Gemini
            interpretation_response = self.gemini_client.generate_response("", interpretation_prompt)
            
            try:
                # Parse the JSON response
                query_params = json.loads(interpretation_response)
                logger.info(f"Query interpretation: {query_params}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing query interpretation: {str(e)}")
                logger.error(f"Raw interpretation: {interpretation_response}")
                # Fallback to a simpler interpretation
                query_params = {
                    "query_type": "other",
                    "time_period": "not_specified",
                    "status_filter": "not_specified",
                    "priority_filter": "not_specified",
                    "requested_analysis": "not_specified",
                    "specific_incident_id": None
                }
            
            # Now fetch the appropriate data based on the interpretation
            incidents = []
            
            # Check for specific incident ID request
            if query_params.get("specific_incident_id"):
                incident = self.helix_api.get_incident_by_id(query_params["specific_incident_id"])
                if incident:
                    incidents = [incident]
            
            # Handle time-based queries
            elif query_params.get("time_period") in ["yesterday", "last_week", "last_month", "specific_date", "date_range"]:
                end_date = datetime.now()
                
                if query_params["time_period"] == "yesterday":
                    start_date = end_date - timedelta(days=1)
                elif query_params["time_period"] == "last_week":
                    start_date = end_date - timedelta(days=7)
                elif query_params["time_period"] == "last_month":
                    start_date = end_date - timedelta(days=30)
                elif query_params["time_period"] == "specific_date":
                    if query_params.get("specific_date"):
                        try:
                            specific_date = datetime.strptime(query_params["specific_date"], "%Y-%m-%d")
                            start_date = specific_date
                            end_date = specific_date + timedelta(days=1)
                        except ValueError:
                            start_date = end_date - timedelta(days=1)
                    else:
                        start_date = end_date - timedelta(days=1)
                elif query_params["time_period"] == "date_range":
                    if query_params.get("date_range_start") and query_params.get("date_range_end"):
                        try:
                            start_date = datetime.strptime(query_params["date_range_start"], "%Y-%m-%d")
                            end_date = datetime.strptime(query_params["date_range_end"], "%Y-%m-%d") + timedelta(days=1)
                        except ValueError:
                            start_date = end_date - timedelta(days=7)
                    else:
                        start_date = end_date - timedelta(days=7)
                
                logger.info(f"Fetching incidents from {start_date} to {end_date}")
                incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
            
            # Handle status filter
            if query_params.get("status_filter") and query_params["status_filter"] not in ["all", "not_specified"]:
                if not incidents:  # If we haven't fetched incidents yet
                    incidents = self.helix_api.get_incidents_by_status(query_params["status_filter"])
                else:  # Filter the already fetched incidents
                    incidents = [inc for inc in incidents if inc.get("Status", "").lower() == query_params["status_filter"].lower()]
            
            # Handle priority filter
            if query_params.get("priority_filter") and query_params["priority_filter"] not in ["all", "not_specified"]:
                if not incidents:  # If we haven't fetched incidents yet
                    incidents = self.helix_api.get_incidents_by_priority(query_params["priority_filter"])
                else:  # Filter the already fetched incidents
                    incidents = [inc for inc in incidents if inc.get("Priority", "").lower() == query_params["priority_filter"].lower()]
            
            # Handle group filter
            if query_params.get("group_filter"):
                if not incidents:  # If we haven't fetched incidents yet
                    incidents = self.helix_api.get_incidents_by_support_group(query_params["group_filter"])
                else:  # Filter the already fetched incidents
                    incidents = [inc for inc in incidents if query_params["group_filter"].lower() in inc.get("Support Group Name", "").lower()]
            
            # Handle assignee filter
            if query_params.get("assignee_filter"):
                if not incidents:  # If we haven't fetched incidents yet
                    incidents = self.helix_api.get_incidents_by_assignee(query_params["assignee_filter"])
                else:  # Filter the already fetched incidents
                    incidents = [inc for inc in incidents if query_params["assignee_filter"].lower() in inc.get("Request Assignee", "").lower()]
            
            # Handle search terms
            if query_params.get("search_terms"):
                if not incidents:  # If we haven't fetched incidents yet
                    incidents = self.helix_api.search_incidents(query_params["search_terms"])
                else:  # Filter the already fetched incidents
                    search_term = query_params["search_terms"].lower()
                    incidents = [inc for inc in incidents if 
                                search_term in str(inc.get("Summary", "")).lower() or
                                search_term in str(inc.get("Description", "")).lower()]
            
            # If we still don't have any incidents, get a default set
            if not incidents:
                # Get all incidents from the last 7 days as a default
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
            
            # Prepare the incidents for analysis
            incidents_json = json.dumps(incidents, indent=2)
            
            # Construct the prompt for Gemini based on the query type and requested analysis
            if query_params.get("query_type") == "incident_details" and query_params.get("specific_incident_id"):
                analysis_prompt = f"""
                The user has asked for details about incident {query_params["specific_incident_id"]}.
                
                Here is the incident data:
                ```json
                {incidents_json}
                ```
                
                Please provide a detailed summary of this incident, including its current status, priority, assignment, and other relevant details.
                Format your response in a professional and clear manner suitable for an IT service management context.
                
                Original query: "{query}"
                """
            elif query_params.get("query_type") == "incident_statistics" or query_params.get("requested_analysis") == "categorization":
                analysis_prompt = f"""
                The user has asked for statistics or categorization of incidents.
                
                Here is the incident data:
                ```json
                {incidents_json}
                ```
                
                Please analyze these incidents and provide statistics and categorization. Include:
                1. Count of incidents by status
                2. Count of incidents by priority
                3. Count of incidents by support group
                4. Any other relevant categorizations based on the user's query
                
                Format your response with appropriate tables and summaries. Be professional and clear in your explanation.
                
                Original query: "{query}"
                """
            elif query_params.get("requested_analysis") == "trends":
                analysis_prompt = f"""
                The user has asked for trend analysis of incidents.
                
                Here is the incident data:
                ```json
                {incidents_json}
                ```
                
                Please analyze these incidents for trends over time. Consider:
                1. Changes in incident volume over time
                2. Patterns in incident categories or types
                3. Shifts in priority distribution
                4. Changes in resolution time or assignment patterns
                
                Format your response with clear explanations of the trends identified. Be professional and data-driven in your analysis.
                Use tables and lists to clearly present your findings.
                
                Original query: "{query}"
                """
            else:
                # General incident list and analysis
                analysis_prompt = f"""
                The user has asked the following question about incidents:
                "{query}"
                
                Here is the incident data:
                ```json
                {incidents_json}
                ```
                
                Please analyze these incidents and provide a comprehensive response that answers the user's query.
                Format your response appropriately with tables, summaries, or lists as needed. Be professional and clear in your explanation.
                
                Be sure to organize your response in a way that directly addresses what the user asked for. Your response format should match the needs of the query.
                
                If the user's question can't be answered with the provided data, please explain why and suggest alternative approaches.
                """
            
            # Get the final response from Gemini
            final_response = self.gemini_client.generate_response(self.incident_analysis_instructions, analysis_prompt)
            
            return final_response
            
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
        """
        Process a user query and return a response.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The response to the query
        """
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
    print("- Categorize incidents from the last month by status and priority")
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