#!/usr/bin/env python3
"""
Advanced BMC Helix + Gemini Chatbot Integration

This enhanced integration combines BMC Helix with Gemini AI to create a powerful
chatbot for IT service management that can handle various query types and formats.

Features:
- WebUI interface with Flask
- Advanced query processing
- Rich response formatting with charts/tables
- Extended incident management capabilities
- Session management and history tracking
"""

import os
import sys
import logging
import json
import re
import time
import requests
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from tabulate import tabulate
from flask import Flask, request, jsonify, render_template, session

# Google Gemini imports
from google import genai
from google.api_core.exceptions import GoogleAPICallError
from google.genai import types

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_helix_gemini_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AdvancedHelixGeminiChatbot")

# Configuration (Hardcoded for testing as per requirements)
PROJECT_ID = "prj-dv-cws-4363"
REGION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"

# BMC Helix configuration
BMC_SERVER = "cmegroup-restapi.onbmc.com"
BMC_USERNAME = "username"  # Replace with actual username
BMC_PASSWORD = "password"  # Replace with actual password

class BMCHelixAPI:
    """Enhanced client for interacting with the BMC Helix API."""
    
    def __init__(self, server=BMC_SERVER, username=BMC_USERNAME, password=BMC_PASSWORD):
        """Initialize the BMC Helix API client."""
        self.server = server
        self.username = username
        self.password = password
        self.token = None
        self.headers = None
        self.session_id = str(uuid.uuid4())
        self.last_login_time = None
        self.token_expiry = 3600  # Typical token expiry in seconds (adjust based on your system)
    
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
                self.last_login_time = datetime.now()
                logger.info("Login successful")
                return True
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            return False
    
    def check_token_expiry(self) -> bool:
        """Check if token is expired and refresh if needed."""
        if not self.token or not self.last_login_time:
            return self.login()
            
        # Check if token is about to expire (give 5 minute buffer)
        time_since_login = (datetime.now() - self.last_login_time).total_seconds()
        if time_since_login > (self.token_expiry - 300):
            logger.info("Token approaching expiry, refreshing...")
            return self.login()
            
        return True
    
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
                self.last_login_time = None
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            return False
    
    def get_incidents(self, query_params=None, max_results=100) -> List[Dict]:
        """
        Get incidents from BMC Helix with pagination support.
        
        Args:
            query_params: Query parameters to filter incidents
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: List of incidents
        """
        self.check_token_expiry()
        
        if not self.token:
            logger.warning("Not logged in, attempting to login")
            if not self.login():
                return []
        
        default_fields = "Status,Summary,Support Group Name,Request Assignee,Submitter,Work Order ID,Request Manager,Incident Number,Description,Status,Owner,Submitter,Impact,Owner Group,Submit Date,Assigned Group,Priority,Environment"
        
        url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help%20Desk"
        
        params = {
            'fields': default_fields,
            'limit': min(max_results, 1000),  # API limit per request
        }
        
        if query_params:
            for key, value in query_params.items():
                if key.startswith('&'):
                    params[key[1:]] = value
                else:
                    params[key] = value
        
        all_incidents = []
        offset = 0
        
        try:
            while True:
                if offset > 0:
                    params['offset'] = offset
                
                r = requests.get(url, headers=self.headers, params=params, verify=False)
                
                if r.status_code == 200:
                    data = r.json()
                    entries = data.get('entries', [])
                    
                    if not entries:
                        break
                        
                    # Normalize the data structure
                    for entry in entries:
                        incident = {}
                        for key, value in entry.get('values', {}).items():
                            incident[key] = value
                        all_incidents.append(incident)
                    
                    # Check if we've reached max_results or end of data
                    if len(all_incidents) >= max_results or len(entries) < params.get('limit', 1000):
                        break
                        
                    offset += len(entries)
                    
                elif r.status_code == 401:
                    logger.warning("Token expired, attempting to re-login")
                    if self.login():
                        # Retry this batch
                        continue
                    else:
                        break
                else:
                    logger.error(f"Failed to get incidents with status code: {r.status_code}")
                    logger.error(f"Response: {r.text}")
                    break
                    
            logger.info(f"Retrieved {len(all_incidents)} incidents total")
            return all_incidents[:max_results]
            
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
        """Get incidents with a specific status."""
        query_params = {
            'q': f"'Status'=\"{status}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_support_group(self, support_group: str) -> List[Dict]:
        """Get incidents assigned to a specific support group."""
        query_params = {
            'q': f"'Support Group Name'=\"{support_group}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_priority(self, priority: str) -> List[Dict]:
        """Get incidents with a specific priority."""
        query_params = {
            'q': f"'Priority'=\"{priority}\""
        }
        
        return self.get_incidents(query_params)
    
    def get_incidents_by_assignee(self, assignee: str) -> List[Dict]:
        """Get incidents assigned to a specific user."""
        query_params = {
            'q': f"'Request Assignee'=\"{assignee}\""
        }
        
        return self.get_incidents(query_params)
    
    def search_incidents(self, search_term: str) -> List[Dict]:
        """Search for incidents containing a specific term in summary or description."""
        query_params = {
            'q': f"'Summary' LIKE \"%{search_term}%\" OR 'Description' LIKE \"%{search_term}%\""
        }
        
        return self.get_incidents(query_params)
    
    def execute_custom_query(self, query: str) -> List[Dict]:
        """Execute a custom query against the BMC Helix API."""
        query_params = {
            'q': query
        }
        
        return self.get_incidents(query_params)
    
    def create_incident(self, incident_data: Dict) -> Optional[str]:
        """
        Create a new incident in BMC Helix.
        
        Args:
            incident_data (Dict): Incident data to create
            
        Returns:
            Optional[str]: Incident ID if successful, None otherwise
        """
        self.check_token_expiry()
        
        url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help%20Desk"
        
        try:
            r = requests.post(url, headers=self.headers, json={'values': incident_data}, verify=False)
            
            if r.status_code == 201:
                location = r.headers.get('Location', '')
                # Extract incident ID from location
                incident_id = location.split('/')[-1] if location else None
                logger.info(f"Created incident with ID: {incident_id}")
                return incident_id
            elif r.status_code == 401:
                logger.warning("Token expired, attempting to re-login")
                if self.login():
                    return self.create_incident(incident_data)
                return None
            else:
                logger.error(f"Failed to create incident with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Error creating incident: {str(e)}")
            return None
    
    def update_incident(self, incident_id: str, update_data: Dict) -> bool:
        """
        Update an existing incident in BMC Helix.
        
        Args:
            incident_id (str): Incident ID to update
            update_data (Dict): Data to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.check_token_expiry()
        
        url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help%20Desk/{incident_id}"
        
        try:
            r = requests.put(url, headers=self.headers, json={'values': update_data}, verify=False)
            
            if r.status_code == 204:
                logger.info(f"Updated incident with ID: {incident_id}")
                return True
            elif r.status_code == 401:
                logger.warning("Token expired, attempting to re-login")
                if self.login():
                    return self.update_incident(incident_id, update_data)
                return False
            else:
                logger.error(f"Failed to update incident with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Error updating incident: {str(e)}")
            return False


class EnhancedGeminiClient:
    """Enhanced client for interacting with Google's Gemini AI model."""
    
    def __init__(self, project_id=PROJECT_ID, region=REGION, model_name=MODEL_NAME):
        """Initialize the Gemini client."""
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize the Gemini client."""
        try:
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.region,
            )
            logger.info(f"Initialized Gemini client for model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.client = None
    
    def generate_response(self, system_instructions: str, prompt: str, 
                         temperature: float = 0.2, max_tokens: int = 8192) -> str:
        """
        Generate a response from the Gemini model with customizable parameters.
        
        Args:
            system_instructions (str): System instructions to guide the model
            prompt (str): The prompt to send to the model
            temperature (float): Temperature parameter (0.0-1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: The generated response
        """
        if not self.client:
            logger.error("Gemini client not initialized")
            return "Error: Unable to connect to Gemini AI"
        
        logger.info(f"Generating response with temp={temperature} for prompt: {prompt[:100]}...")
        
        try:
            # Create system instruction if provided
            system_instruction = None
            if system_instructions:
                system_instruction = types.Part.from_text(text=system_instructions)
            
            # Configure generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    )
                ]
            )
            
            # Stream the response
            response_text = ""
            
            # Create request with or without system instruction
            if system_instruction:
                for chunk in self.client.generate_content_stream(
                    model=self.model_name,
                    contents=[system_instruction, prompt],
                    generation_config=generate_content_config,
                ):
                    if not chunk.candidates or not chunk.candidates[0].content.parts:
                        continue
                    response_text += chunk.text
            else:
                for chunk in self.client.generate_content_stream(
                    model=self.model_name,
                    contents=[prompt],
                    generation_config=generate_content_config,
                ):
                    if not chunk.candidates or not chunk.candidates[0].content.parts:
                        continue
                    response_text += chunk.text
            
            logger.info(f"Generated response of length {len(response_text)}")
            return response_text
        
        except GoogleAPICallError as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        except Exception as e:
            error_msg = f"Unexpected error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class AdvancedQueryProcessor:
    """Advanced processing and interpretation of user queries."""
    
    def __init__(self, helix_api: BMCHelixAPI, gemini_client: EnhancedGeminiClient):
        """Initialize the advanced query processor."""
        self.helix_api = helix_api
        self.gemini_client = gemini_client
        
        # Define comprehensive system instructions
        self.system_instructions = """
        You are a professional and friendly IT service management assistant specializing in BMC Helix.
        You help IT teams analyze and understand their incident data to improve service delivery.
        
        When responding:
        1. Be professional, courteous, and concise
        2. Structure your responses with appropriate formatting (tables, lists, etc.)
        3. Provide data-driven insights when possible
        4. Always maintain a helpful and solution-oriented tone
        
        If you cannot answer a question due to data limitations, clearly explain why and suggest alternative approaches.
        """
        
        # Instructions for analyzing incident data
        self.incident_analysis_instructions = """
        You are an expert IT service management analyst specializing in BMC Helix incidents.
        
        When analyzing BMC Helix incident data:
        1. Consider all relevant fields including Status, Priority, Support Group, Assignee
        2. Structure your response with clear sections: Summary, Analysis, and Recommendations
        3. Use tables and formatted text to present information clearly
        4. Highlight notable patterns or outliers in the data
        5. Suggest potential actions based on your analysis when appropriate
        
        Present your findings in a professional format suitable for IT managers and support teams.
        Include relevant statistics and categorize information logically based on the query.
        
        Focus on delivering actionable insights that help teams improve their service delivery.
        """
        
        # Instructions for handling advanced analytics requests
        self.advanced_analytics_instructions = """
        You are an advanced IT analytics expert specializing in BMC Helix data.
        
        When providing advanced analytics:
        1. Use statistical analysis where appropriate
        2. Identify patterns, trends, and correlations in the data
        3. Present visual-friendly descriptions of data distributions
        4. Compare current data with historical norms when context is provided
        5. Suggest potential root causes for observed patterns
        
        Present your analysis in a clear, structured format with appropriate sections.
        Use tables, bulleted lists, and other formatting to enhance readability.
        
        Focus on extracting meaningful insights that can drive operational improvements.
        """
    
    def detect_query_intent(self, query: str) -> Dict:
        """
        Use Gemini to detect the intent and parameters of a user query.
        
        Args:
            query (str): The user's query
            
        Returns:
            Dict: Query intent parameters
        """
        interpretation_prompt = f"""
        Analyze this query about BMC Helix incidents to determine the exact information being requested:
        
        Query: "{query}"
        
        Provide a structured JSON analysis with these fields:
        {{
            "query_intent": "list" | "details" | "statistics" | "categorization" | "trends" | "search" | "other",
            "time_period": "today" | "yesterday" | "this_week" | "last_week" | "this_month" | "last_month" | "specific_date" | "date_range" | "not_specified",
            "time_specifics": {{
                "specific_date": "YYYY-MM-DD or null",
                "date_range_start": "YYYY-MM-DD or null",
                "date_range_end": "YYYY-MM-DD or null"
            }},
            "filters": {{
                "status": "all" | "open" | "closed" | "resolved" | "in_progress" | "null",
                "priority": "critical" | "high" | "medium" | "low" | "null",
                "support_group": "string or null",
                "assignee": "string or null"
            }},
            "analysis_type": "summary" | "detailed" | "comparative" | "predictive" | "null",
            "incident_id": "string or null if not about a specific incident",
            "search_terms": "string or null if not a search query"
        }}
        
        Return ONLY valid JSON without additional text, comments or explanation.
        """
        
        interpretation_response = self.gemini_client.generate_response("", interpretation_prompt, temperature=0.1)
        
        try:
            # Parse the JSON response
            intent_params = json.loads(interpretation_response)
            logger.info(f"Query intent detection: {intent_params}")
            return intent_params
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing query intent: {str(e)}")
            logger.error(f"Raw interpretation: {interpretation_response}")
            # Fallback to a basic intent
            return {
                "query_intent": "other",
                "time_period": "not_specified",
                "filters": {
                    "status": "null",
                    "priority": "null",
                    "support_group": None,
                    "assignee": None
                },
                "analysis_type": "null",
                "incident_id": None,
                "search_terms": None
            }
    
    def fetch_incidents_for_query(self, intent_params: Dict) -> List[Dict]:
        """
        Fetch incidents based on query intent parameters.
        
        Args:
            intent_params: Query intent parameters
            
        Returns:
            List[Dict]: Matching incidents
        """
        # Check for specific incident ID
        if intent_params.get("incident_id"):
            incident = self.helix_api.get_incident_by_id(intent_params["incident_id"])
            if incident:
                return [incident]
            return []
        
        # Handle time-based queries
        end_date = datetime.now()
        start_date = None
        
        time_period = intent_params.get("time_period", "not_specified")
        
        if time_period == "today":
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "yesterday":
            start_date = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "this_week":
            # Start from Monday of current week
            today = datetime.now()
            start_date = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "last_week":
            # Last week Monday to Sunday
            today = datetime.now()
            start_date = (today - timedelta(days=today.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "this_month":
            # Start from first day of current month
            start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "last_month":
            # Previous month
            today = datetime.now()
            start_date = (today.replace(day=1) - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "specific_date" and intent_params.get("time_specifics", {}).get("specific_date"):
            try:
                specific_date = datetime.strptime(intent_params["time_specifics"]["specific_date"], "%Y-%m-%d")
                start_date = specific_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = specific_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            except ValueError:
                # Fallback to last 7 days
                start_date = end_date - timedelta(days=7)
        elif time_period == "date_range":
            time_specifics = intent_params.get("time_specifics", {})
            try:
                if time_specifics.get("date_range_start") and time_specifics.get("date_range_end"):
                    start_date = datetime.strptime(time_specifics["date_range_start"], "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = datetime.strptime(time_specifics["date_range_end"], "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
            except ValueError:
                # Fallback to last 7 days
                start_date = end_date - timedelta(days=7)
        
        # Default to last 7 days if no time period specified
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        # Fetch incidents by date range as base query
        incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date)
        
        # Apply additional filters
        filters = intent_params.get("filters", {})
        
        # Status filter
        if filters.get("status") and filters["status"] not in ["all", "null"]:
            incidents = [inc for inc in incidents if inc.get("Status", "").lower() == filters["status"].lower()]
        
        # Priority filter
        if filters.get("priority") and filters["priority"] not in ["all", "null"]:
            incidents = [inc for inc in incidents if inc.get("Priority", "").lower() == filters["priority"].lower()]
        
        # Support group filter
        if filters.get("support_group"):
            incidents = [inc for inc in incidents if filters["support_group"].lower() in str(inc.get("Support Group Name", "")).lower()]
        
        # Assignee filter
        if filters.get("assignee"):
            incidents = [inc for inc in incidents if filters["assignee"].lower() in str(inc.get("Request Assignee", "")).lower()]
        
        # Search terms
        if intent_params.get("search_terms"):
            search_term = intent_params["search_terms"].lower()
            incidents = [inc for inc in incidents if 
                        search_term in str(inc.get("Summary", "")).lower() or
                        search_term in str(inc.get("Description", "")).lower()]
        
        return incidents
    
    def enrich_incidents_data(self, incidents: List[Dict]) -> List[Dict]:
        """
        Enrich incident data with additional computed fields.
        
        Args:
            incidents: Raw incident data
            
        Returns:
            List[Dict]: Enriched incident data
        """
        enriched_incidents = []
        
        for incident in incidents:
            # Create a copy of the incident to avoid modifying the original
            enriched = incident.copy()
            
            # Calculate time-based metrics if Submit Date is available
            if "Submit Date" in incident:
                try:
                    submit_date = datetime.strptime(incident["Submit Date"], "%Y-%m-%d %H:%M:%S")
                    
                    # Add day of week
                    enriched["Day_of_Week"] = submit_date.strftime("%A")
                    
                    # Add hour of day
                    enriched["Hour_of_Day"] = submit_date.hour
                    
                    # Calculate age in days (from submit date to now)
                    age_days = (datetime.now() - submit_date).total_seconds() / (60 * 60 * 24)
                    enriched["Age_Days"] = round(age_days, 1)
                    
                except (ValueError, TypeError):
                    # If date parsing fails, skip these enrichments
                    pass
            
            enriched_incidents.append(enriched)
        
        return enriched_incidents
    
    def generate_analytics_prompt(self, query: str, incidents: List[Dict], intent_params: Dict) -> str:
        """
        Generate an appropriate analytics prompt based on query intent.
        
        Args:
            query: Original user query
            incidents: Incident data
            intent_params: Query intent parameters
            
        Returns:
            str: Generated prompt for Gemini
        """
        # Convert incidents to JSON for the prompt
        incidents_json = json.dumps(incidents, indent=2)
        
        # Generate base prompt with query context
        prompt = f"""
        Original user query: "{query}"
        
        I've retrieved {len(incidents)} incidents from BMC Helix based on this query.
        
        Incident data:
        ```json
        {incidents_json}
        ```
        """
        
        # Customize prompt based on query intent
        if intent_params["query_intent"] == "details" and intent_params.get("incident_id"):
            prompt += f"""
            The user wants detailed information about incident {intent_params["incident_id"]}.
            
            Please provide a comprehensive summary of this incident including:
            1. Basic incident information (ID, status, priority, etc.)
            2. Assignment details (support group, assignee)
            3. Timeline information (submitted, last modified)
            4. Complete incident description
            5. Any other relevant details
            
            Format the response as a structured report suitable for IT professionals.
            """
        
        elif intent_params["query_intent"] == "statistics" or intent_params["query_intent"] == "categorization":
            prompt += f"""
            The user wants statistical analysis or categorization of these incidents.
            
            Please provide a comprehensive statistical breakdown including:
            1. Incident counts by status
            2. Incident counts by priority
            3. Top support groups by incident volume
            4. Top incident categories or types (based on summary content)
            
            Present your analysis using tables and clear section headings.
            Include percentages where appropriate.
            
            End with 2-3 key observations about the data that would be valuable to IT management.
            """
        
        elif intent_params["query_intent"] == "trends":
            prompt += f"""
            The user wants trend analysis of these incidents.
            
            Please analyze these incidents for temporal patterns and trends, including:
            1. Volume trends (incidents over time)
            2. Priority distribution trends
            3. Support group assignment patterns
            4. Common themes in incident descriptions
            
            Describe any patterns you observe in clear, concise language.
            Use tables to present numerical trends when appropriate.
            
            End with 2-3 insights about what these trends might indicate about the IT environment.
            """
        
        elif intent_params["query_intent"] == "search":
            prompt += f"""
            The user is searching for incidents related to: "{intent_params.get('search_terms')}"
            
            Please analyze these search results and:
            1. Summarize the key information about these incidents
            2. Group them by relevant categories (status, priority, etc.)
            3. Highlight any patterns or commonalities
            
            Present your findings in a clear, structured format.
            Include a summary table of key incident information.
            """
        
        else:  # Default for list and other queries
            prompt += f"""
            Please analyze these incidents and provide a response that best addresses the user's query.
            
            Include:
            1. A summary of the incident data (total count, status distribution, priority distribution)
            2. A concise table with key incident information
            3. Any notable patterns or insights
            
            Format your response professionally with clear structure and appropriate tables or lists.
            Directly address what the user asked for in their query.
            """
        
        return prompt
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a comprehensive response.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The response to the query
        """
        logger.info(f"Processing advanced query: {query}")
        
        # Detect query intent
        intent_params = self.detect_query_intent(query)
        
        # Fetch relevant incidents
        incidents = self.fetch_incidents_for_query(intent_params)
        
        # If no incidents found, handle gracefully
        if not incidents:
            no_data_prompt = f"""
            I tried to find incident data matching this query: "{query}"
            
            However, I couldn't find any incidents matching the criteria in the BMC Helix system.
            
            Please provide a helpful response explaining that no matching incidents were found,
            and suggesting some alternative queries or approaches the user might try instead.
            Keep your response professional and helpful.
            """
            
            return self.gemini_client.generate_response(self.system_instructions, no_data_prompt)
        
        # Enrich incident data with computed fields
        enriched_incidents = self.enrich_incidents_data(incidents)
        
        # Generate analytics prompt
        analytics_prompt = self.generate_analytics_prompt(query, enriched_incidents, intent_params)
        
        # Choose appropriate instructions based on query intent
        if intent_params["query_intent"] in ["statistics", "categorization", "trends"]:
            system_instructions = self.advanced_analytics_instructions
        else:
            system_instructions = self.incident_analysis_instructions
        
        # Generate response
        response = self.gemini_client.generate_response(system_instructions, analytics_prompt)
        
        return response


class AdvancedHelixGeminiChatbot:
    """Advanced chatbot integrating BMC Helix with Gemini AI."""
    
    def __init__(self):
        """Initialize the advanced chatbot."""
        self.helix_api = BMCHelixAPI()
        self.gemini_client = EnhancedGeminiClient()
        self.query_processor = AdvancedQueryProcessor(self.helix_api, self.gemini_client)
        
        # Try to login to BMC Helix
        login_successful = self.helix_api.login()
        if not login_successful:
            logger.error("Failed to login to BMC Helix. Chatbot may not function correctly.")
        
        # Store conversation history
        self.conversation_history = []
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The response to the query
        """
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Process the query
        response = self.query_processor.process_query(query)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def close(self):
        """Clean up resources and logout."""
        self.helix_api.logout()


# Flask web application for chatbot interface
app = Flask(__name__)
app.secret_key = os.urandom(24)
chatbot = None

@app.route('/')
def index():
    """Render the chatbot web interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat interactions."""
    global chatbot
    
    if chatbot is None:
        chatbot = AdvancedHelixGeminiChatbot()
    
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Empty query'})
    
    try:
        response = chatbot.process_query(query)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"})

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the chatbot conversation."""
    global chatbot
    
    if chatbot:
        chatbot.close()
        chatbot = AdvancedHelixGeminiChatbot()
    
    return jsonify({'status': 'reset successful'})

def interactive_mode():
    """Run the chatbot in interactive command-line mode."""
    print("="*50)
    print("Advanced BMC Helix + Gemini Chatbot")
    print("="*50)
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Example queries:")
    print("- What incidents were created yesterday?")
    print("- Show me high priority incidents from last week")
    print("- Categorize incidents from the last month by status and priority")
    print("- What are the current trends in incident volume?")
    print("- Give me details about incident INC123456")
    print("-"*50)
    
    chatbot = AdvancedHelixGeminiChatbot()
    
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

def web_mode(host='0.0.0.0', port=5000):
    """Run the chatbot as a web application."""
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BMC Helix + Gemini Chatbot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            .chat-container {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: 80vh;
            }
            .header {
                background-color: #0078d4;
                color: white;
                padding: 15px 20px;
                text-align: center;
            }
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
            }
            .input-area {
                display: flex;
                padding: 10px;
                border-top: 1px solid #e0e0e0;
                background-color: #f9f9f9;
            }
            #message-input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-right: 10px;
            }
            button {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #005a9e;
            }
            .message {
                margin-bottom: 15px;
                padding: 10px 15px;
                border-radius: 5px;
                max-width: 80%;
            }
            .user-message {
                background-color: #e3f2fd;
                margin-left: auto;
                text-align: right;
            }
            .bot-message {
                background-color: #f1f1f1;
                margin-right: auto;
            }
            .loading {
                display: inline-block;
                margin-left: 10px;
            }
            .loading span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #0078d4;
                margin: 0 3px;
                animation: loading 1s infinite;
            }
            .loading span:nth-child(2) {
                animation-delay: 0.2s;
            }
            .loading span:nth-child(3) {
                animation-delay: 0.4s;
            }
            pre {
                background-color: #f7f7f7;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }
            table, th, td {
                border: 1px solid #ddd;
            }
            th, td {
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            @keyframes loading {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>BMC Helix + Gemini Chatbot</h1>
            </div>
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    <div class="message bot-message">
                        Hello! I'm your BMC Helix assistant. I can help you with incident information, statistics, and analysis. What would you like to know?
                    </div>
                </div>
                <div class="input-area">
                    <input type="text" id="message-input" placeholder="Ask about incidents, e.g., 'What incidents were created yesterday?'" />
                    <button id="send-button">Send</button>
                    <button id="reset-button">Reset</button>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const chatMessages = document.getElementById('chat-messages');
                const messageInput = document.getElementById('message-input');
                const sendButton = document.getElementById('send-button');
                const resetButton = document.getElementById('reset-button');
                
                function addMessage(content, isUser) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
                    
                    // Convert markdown-style tables to HTML tables
                    let formattedContent = content;
                    
                    // Format markdown tables
                    const tableRegex = /\|(.+)\|[\r\n]\|([-]+\|)+[\r\n]((.*\|[\r\n])+)/g;
                    formattedContent = formattedContent.replace(tableRegex, function(match) {
                        const lines = match.split('\\n');
                        let html = '<table>';
                        
                        // Header
                        let headerCells = lines[0].split('|').filter(cell => cell.trim() !== '');
                        html += '<tr>';
                        headerCells.forEach(cell => {
                            html += `<th>${cell.trim()}</th>`;
                        });
                        html += '</tr>';
                        
                        // Skip the separator line (line[1])
                        
                        // Body
                        for (let i = 2; i < lines.length; i++) {
                            if (lines[i].trim() === '') continue;
                            let cells = lines[i].split('|').filter(cell => cell.trim() !== '');
                            html += '<tr>';
                            cells.forEach(cell => {
                                html += `<td>${cell.trim()}</td>`;
                            });
                            html += '</tr>';
                        }
                        
                        html += '</table>';
                        return html;
                    });
                    
                    // Format code blocks
                    formattedContent = formattedContent.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>');
                    
                    // Format bold text
                    formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    
                    // Format italic text
                    formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
                    
                    // Format lists
                    formattedContent = formattedContent.replace(/^\s*[-*]\s+(.*)/gm, '<li>$1</li>');
                    formattedContent = formattedContent.replace(/<li>(.*?)<\/li>/g, '<ul><li>$1</li></ul>');
                    
                    // Format line breaks
                    formattedContent = formattedContent.replace(/\n/g, '<br>');
                    
                    messageDiv.innerHTML = formattedContent;
                    chatMessages.appendChild(messageDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function showLoading() {
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'message bot-message loading-message';
                    loadingDiv.innerHTML = 'Thinking<div class="loading"><span></span><span></span><span></span></div>';
                    chatMessages.appendChild(loadingDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    return loadingDiv;
                }
                
                function removeLoading(loadingDiv) {
                    chatMessages.removeChild(loadingDiv);
                }
                
                function sendMessage() {
                    const message = messageInput.value.trim();
                    if (message === '') return;
                    
                    addMessage(message, true);
                    messageInput.value = '';
                    
                    const loadingDiv = showLoading();
                    
                    fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: message })
                    })
                    .then(response => response.json())
                    .then(data => {
                        removeLoading(loadingDiv);
                        if (data.error) {
                            addMessage('Error: ' + data.error, false);
                        } else {
                            addMessage(data.response, false);
                        }
                    })
                    .catch(error => {
                        removeLoading(loadingDiv);
                        addMessage('Error connecting to server: ' + error, false);
                    });
                }
                
                function resetConversation() {
                    fetch('/api/reset', {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Clear the chat
                        chatMessages.innerHTML = '';
                        addMessage('Hello! I\'m your BMC Helix assistant. I can help you with incident information, statistics, and analysis. What would you like to know?', false);
                    })
                    .catch(error => {
                        addMessage('Error resetting conversation: ' + error, false);
                    });
                }
                
                sendButton.addEventListener('click', sendMessage);
                resetButton.addEventListener('click', resetConversation);
                
                messageInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    print(f"Starting web interface at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced BMC Helix + Gemini Chatbot')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli', help='Run in CLI or web mode')
    parser.add_argument('--host', default='0.0.0.0', help='Host for web mode (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port for web mode (default: 5000)')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        web_mode(host=args.host, port=args.port)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()