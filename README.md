#!/usr/bin/env python3
"""
Simple Remedy Gemini Chatbot - Based on your working check_inc script with added Gemini integration
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
import timeit
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

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-001")
bg_ServerName = "cmegroup-restapi.onbmc.com"

# Global variables to maintain session
bg_UserID = None
bg_Password = None
bg_Token = None
bgTokenType = "AR-JWT"

def doLogin():
    """Login to Remedy and get authentication token."""
    global bg_UserID, bg_Password, bg_Token
    
    if not bg_UserID:
        bg_UserID = input("Enter OnePass: ")
    if not bg_Password:
        bg_Password = getpass.getpass(prompt="Enter Password: ")
    
    myToken = ""
    bgReturnVal = -1
    
    url = f"https://{bg_ServerName}/api/jwt/login"
    payload = {'username': bg_UserID, 'password': bg_Password}
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    
    try:
        r = requests.post(url, data=payload, headers=headers, verify=False)
        
        if r.status_code == 200:
            myToken = r.text
            bgReturnVal = 1
        else:
            print("Failure...")
            print("Status Code: " + str(r.status_code))
            bgReturnVal = -1
        
        bg_Token = myToken
        return bgReturnVal, myToken
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return -1, None

def doLogout(bg_Token):
    """Logout from Remedy and invalidate the token."""
    bgTokenType = "AR-JWT"
    myURL = f"https://{bg_ServerName}/api/jwt/logout"
    myHeaders = {'Authorization': bgTokenType + ' ' + bg_Token}
    
    try:
        myR = requests.post(myURL, headers=myHeaders, verify=False)
        return myR.status_code in [200, 204]
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return False

def doPullData_Query(query, bg_Token, fields=None):
    """
    Pull data from Remedy with a custom query and specific fields.
    
    Args:
        query: The qualified query string to use
        bg_Token: Authentication token
        fields: Specific fields to retrieve (or None for all)
        
    Returns:
        List of incidents or None if error
    """
    bgTokenType = "AR-JWT"
    myURL = f"https://{bg_ServerName}/api/arsys/v1/entry/HPD:Help Desk"
    myHeaders = {'Authorization': bgTokenType + ' ' + bg_Token}
    
    # Default fields if none specified
    if not fields:
        fields = "Assignee,Incident Number,Description,Status,Owner,Submitter,Impact,Owner Group,Submit Date,Assigned Group,Priority,Environment,Summary,Support Group Name,Request Assignee,Work Order ID,Request Manager"
    
    # Build query parameters
    params = {}
    if query:
        params["q"] = query
    params["fields"] = f"values({fields})"
    
    try:
        myR = requests.get(url=myURL, headers=myHeaders, params=params, verify=False)
        
        if myR.status_code != 200:
            print("Status code:", myR.status_code)
            print("Headers:", myR.headers)
            print("Error Response:", myR.json())
            return None
        
        bgData = myR.json()
        return bgData.get('entries', [])
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None

def get_incident_by_id(incident_id, bg_Token):
    """Get a specific incident by ID."""
    query = f"'Incident Number'=\"{incident_id}\""
    return doPullData_Query(query, bg_Token)

def get_incidents_by_date(date_str, bg_Token):
    """Get incidents by date."""
    # Parse the date
    if date_str.lower() == 'today':
        date_obj = datetime.now()
    elif date_str.lower() == 'yesterday':
        date_obj = datetime.now() - timedelta(days=1)
    elif date_str.lower() == 'last week':
        date_obj = datetime.now() - timedelta(days=7)
    elif date_str.lower() == 'last month':
        date_obj = datetime.now() - timedelta(days=30)
    else:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                return None
    
    # Create date range
    start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
    end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
    
    query = f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""
    return doPullData_Query(query, bg_Token)

def get_incidents_by_status(status, bg_Token):
    """Get incidents by status."""
    query = f"'Status'=\"{status}\""
    return doPullData_Query(query, bg_Token)

def get_incidents_by_owner_group(group, bg_Token):
    """Get incidents by owner group."""
    query = f"'Owner Group'=\"{group}\""
    return doPullData_Query(query, bg_Token)

def get_incidents_by_assignee(assignee, bg_Token):
    """Get incidents by assignee."""
    query = f"'Assignee'=\"{assignee}\""
    return doPullData_Query(query, bg_Token)

def get_incidents_by_text_search(text, bg_Token):
    """Search for incidents containing text in Summary or Description."""
    query = f"'Summary' LIKE \"%{text}%\" OR 'Description' LIKE \"%{text}%\""
    return doPullData_Query(query, bg_Token)

def format_incidents_for_display(incidents):
    """Format incidents for display."""
    if not incidents:
        return "No incidents found."
    
    result = []
    for i, incident in enumerate(incidents, 1):
        if "values" not in incident:
            continue
            
        values = incident["values"]
        
        # Create a formatted entry for each incident
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
        
        # Add description if available (truncated if too long)
        description = values.get('Description')
        if description:
            if len(description) > 200:
                description = description[:200] + "..."
            incident_info.append(f"Description: {description}")
        
        result.append("\n".join(incident_info))
    
    return "\n\n".join(result)

def get_incident_details(incident_id, values):
    """Format detailed incident information."""
    result = [f"Details for Incident {incident_id}:"]
    
    # Add all fields in a more readable format
    for key, value in values.items():
        result.append(f"{key}: {value}")
    
    return "\n".join(result)

class GeminiClient:
    """Client for Vertex AI Gemini model interactions."""
    def __init__(self, project_id=PROJECT_ID, region=REGION, model_name=MODEL_NAME):
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
    
    def generate_response(self, prompt, temperature=0.2):
        """Generate a response using the Gemini model."""
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
            
            # Generate response
            response = self.model.generate_content(
                prompt,
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

def analyze_query(query, gemini):
    """
    Use Gemini to understand the query intent and extract parameters.
    
    Returns a dictionary with intent and parameters.
    """
    analysis_prompt = f"""
    Analyze this query about BMC Remedy incidents and identify the intent and parameters.
    Return a JSON object ONLY (no other text) with the format:
    
    {{
        "intent": "incident_id / incidents_by_date / incidents_by_status / incidents_by_assignee / incidents_by_group / text_search",
        "parameters": {{
            "incident_id": "extracted incident ID",
            "date": "extracted date (today, yesterday, 2023-04-14, etc.)",
            "status": "extracted status (Open, Closed, Resolved, etc.)",
            "assignee": "extracted assignee name",
            "group": "extracted group name",
            "search_text": "extracted search text",
        }}
    }}
    
    Fill only the relevant parameters based on the intent. Leave others empty.
    
    The query is: {query}
    """
    
    try:
        response = gemini.generate_response(analysis_prompt, temperature=0.1)
        
        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        elif response.strip().startswith('{') and response.strip().endswith('}'):
            # Already JSON without code block
            response = response.strip()
        
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        # Default fallback intent
        return {
            "intent": "text_search",
            "parameters": {
                "search_text": query
            }
        }

def execute_query(intent_data, bg_Token):
    """Execute a query based on intent and parameters."""
    intent = intent_data.get("intent", "")
    params = intent_data.get("parameters", {})
    
    if intent == "incident_id":
        incident_id = params.get("incident_id", "")
        if not incident_id:
            return None, "No incident ID provided."
        result = get_incident_by_id(incident_id, bg_Token)
        return result, f"Incident {incident_id}"
    
    elif intent == "incidents_by_date":
        date_str = params.get("date", "")
        if not date_str:
            return None, "No date provided."
        result = get_incidents_by_date(date_str, bg_Token)
        return result, f"Incidents from {date_str}"
    
    elif intent == "incidents_by_status":
        status = params.get("status", "")
        if not status:
            return None, "No status provided."
        result = get_incidents_by_status(status, bg_Token)
        return result, f"Incidents with status '{status}'"
    
    elif intent == "incidents_by_assignee":
        assignee = params.get("assignee", "")
        if not assignee:
            return None, "No assignee provided."
        result = get_incidents_by_assignee(assignee, bg_Token)
        return result, f"Incidents assigned to '{assignee}'"
    
    elif intent == "incidents_by_group":
        group = params.get("group", "")
        if not group:
            return None, "No group provided."
        result = get_incidents_by_owner_group(group, bg_Token)
        return result, f"Incidents owned by group '{group}'"
    
    elif intent == "text_search":
        search_text = params.get("search_text", "")
        if not search_text:
            return None, "No search text provided."
        result = get_incidents_by_text_search(search_text, bg_Token)
        return result, f"Incidents matching '{search_text}'"
    
    else:
        return None, f"Unknown intent: {intent}"

def format_response(incidents, query_description, gemini):
    """Format the final response with Gemini's help."""
    if not incidents:
        return f"No results found for {query_description}."
    
    # First, create a structured representation of the incidents
    formatted_incidents = format_incidents_for_display(incidents)
    
    # Then, use Gemini to create a natural language summary
    summary_prompt = f"""
    Based on the following incident data from Remedy, create a concise and professional summary.
    
    QUERY: {query_description}
    NUMBER OF RESULTS: {len(incidents)}
    
    INCIDENT DATA:
    {formatted_incidents}
    
    Present a clear, concise summary of these incidents. Include relevant statistics (number of incidents by status, priority, etc.)
    and any notable patterns. Format your response appropriately with paragraph breaks and bullet points as needed.
    
    For multiple incidents, start with a high-level overview, then mention the most important/critical incidents specifically.
    For a single incident, provide a more detailed description with all relevant information.
    
    Include the specific incident IDs in your response.
    """
    
    return gemini.generate_response(summary_prompt)

def run_chatbot():
    """Run the Remedy chatbot."""
    print("\n" + "=" * 50)
    print("Remedy Assistant Chatbot")
    print("Ask me about incidents in your Remedy system")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50 + "\n")
    
    # Initialize Gemini
    gemini = GeminiClient()
    
    # Login to Remedy
    status, token = doLogin()
    if status != 1:
        print("Failed to login to Remedy. Please check your credentials.")
        return
    
    print("Successfully logged in to Remedy!")
    
    try:
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using Remedy Assistant. Goodbye!")
                break
            
            start = timeit.default_timer()
            
            # Analyze the query
            print("Processing your query...")
            intent_data = analyze_query(query, gemini)
            logger.info(f"Query intent: {intent_data['intent']}")
            
            # Execute the query
            incidents, query_description = execute_query(intent_data, bg_Token)
            
            # Format and present the response
            if incidents:
                print(f"\nFound {len(incidents)} incidents for {query_description}.")
                response = format_response(incidents, query_description, gemini)
            else:
                response = f"No incidents found for {query_description}. Please try a different query."
                if intent_data['intent'] == 'incident_id':
                    # Special case for incident ID not found
                    incident_id = intent_data['parameters'].get('incident_id', '')
                    response = f"Incident {incident_id} was not found in the system. Please check the ID and try again."
            
            stop = timeit.default_timer()
            print(f"\nAssistant (took {stop-start:.2f}s):")
            print(response)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    finally:
        # Logout when done
        print("\nLogging out from Remedy...")
        if bg_Token:
            doLogout(bg_Token)
            print("Logout successful")

if __name__ == "__main__":
    run_chatbot()
