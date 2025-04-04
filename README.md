#!/usr/bin/env python3
"""
COPPER View-to-API Mapper
--------------------------
Extracts database view definitions and API endpoints from Confluence content,
creates mappings between them, and provides a query interface for users.
"""

import logging
import os
import sys
import json
import re
import time
import concurrent.futures
from collections import defaultdict
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
        logging.FileHandler("copper_mapper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CopperMapper")

# Configuration (same as before)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://your-confluence-instance.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "xyz")

# Define the data structures to hold our views and API endpoints

class DatabaseView:
    """Represents a database view definition with its columns and properties."""
    
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.columns = []  # List of ViewColumn objects
        self.primary_keys = []  # List of column names that are primary keys
        self.relationships = []  # List of Relationship objects
        self.source_page = None  # Confluence page where this view was defined
        
    def add_column(self, column):
        """Add a column to this view."""
        self.columns.append(column)
        
    def add_relationship(self, relationship):
        """Add a relationship to this view."""
        self.relationships.append(relationship)
        
    def mark_primary_key(self, column_name):
        """Mark a column as a primary key."""
        if column_name not in self.primary_keys:
            self.primary_keys.append(column_name)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "columns": [col.to_dict() for col in self.columns],
            "primary_keys": self.primary_keys,
            "relationships": [rel.to_dict() for rel in self.relationships],
            "source_page": self.source_page.get("title", "Unknown") if self.source_page else "Unknown"
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.primary_keys = data.get("primary_keys", [])
        
        # Columns
        self.columns = []
        for col_data in data.get("columns", []):
            column = ViewColumn("", "")
            column.from_dict(col_data)
            self.columns.append(column)
            
        # Relationships
        self.relationships = []
        for rel_data in data.get("relationships", []):
            relationship = Relationship("", "", "", "")
            relationship.from_dict(rel_data)
            self.relationships.append(relationship)
        
        return self

class ViewColumn:
    """Represents a column in a database view."""
    
    def __init__(self, name, data_type, description="", nullable=True):
        self.name = name
        self.data_type = data_type
        self.description = description
        self.nullable = nullable
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description,
            "nullable": self.nullable
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.name = data.get("name", "")
        self.data_type = data.get("data_type", "")
        self.description = data.get("description", "")
        self.nullable = data.get("nullable", True)
        return self

class Relationship:
    """Represents a relationship between database views."""
    
    def __init__(self, from_view, from_column, to_view, to_column, relationship_type="many-to-one"):
        self.from_view = from_view
        self.from_column = from_column
        self.to_view = to_view
        self.to_column = to_column
        self.relationship_type = relationship_type
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "from_view": self.from_view,
            "from_column": self.from_column,
            "to_view": self.to_view,
            "to_column": self.to_column,
            "relationship_type": self.relationship_type
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.from_view = data.get("from_view", "")
        self.from_column = data.get("from_column", "")
        self.to_view = data.get("to_view", "")
        self.to_column = data.get("to_column", "")
        self.relationship_type = data.get("relationship_type", "many-to-one")
        return self

class ApiEndpoint:
    """Represents a REST API endpoint definition."""
    
    def __init__(self, path, method="GET", description=""):
        self.path = path
        self.method = method
        self.description = description
        self.parameters = []  # List of ApiParameter objects
        self.request_body = None  # ApiRequestBody object
        self.response = None  # ApiResponse object
        self.source_page = None  # Confluence page where this endpoint was defined
        
    def add_parameter(self, parameter):
        """Add a parameter to this endpoint."""
        self.parameters.append(parameter)
        
    def set_request_body(self, request_body):
        """Set the request body for this endpoint."""
        self.request_body = request_body
        
    def set_response(self, response):
        """Set the response for this endpoint."""
        self.response = response
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "method": self.method,
            "description": self.description,
            "parameters": [param.to_dict() for param in self.parameters],
            "request_body": self.request_body.to_dict() if self.request_body else None,
            "response": self.response.to_dict() if self.response else None,
            "source_page": self.source_page.get("title", "Unknown") if self.source_page else "Unknown"
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.path = data.get("path", "")
        self.method = data.get("method", "GET")
        self.description = data.get("description", "")
        
        # Parameters
        self.parameters = []
        for param_data in data.get("parameters", []):
            param = ApiParameter("", "")
            param.from_dict(param_data)
            self.parameters.append(param)
            
        # Request body
        if data.get("request_body"):
            self.request_body = ApiRequestBody()
            self.request_body.from_dict(data["request_body"])
        else:
            self.request_body = None
            
        # Response
        if data.get("response"):
            self.response = ApiResponse()
            self.response.from_dict(data["response"])
        else:
            self.response = None
            
        return self

class ApiParameter:
    """Represents a parameter for an API endpoint."""
    
    def __init__(self, name, data_type, description="", required=False, location="query"):
        self.name = name
        self.data_type = data_type
        self.description = description
        self.required = required
        self.location = location  # query, path, header, cookie
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description,
            "required": self.required,
            "location": self.location
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.name = data.get("name", "")
        self.data_type = data.get("data_type", "")
        self.description = data.get("description", "")
        self.required = data.get("required", False)
        self.location = data.get("location", "query")
        return self

class ApiRequestBody:
    """Represents a request body for an API endpoint."""
    
    def __init__(self, content_type="application/json", fields=None):
        self.content_type = content_type
        self.fields = fields or []  # List of ApiField objects
        
    def add_field(self, field):
        """Add a field to this request body."""
        self.fields.append(field)
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "content_type": self.content_type,
            "fields": [field.to_dict() for field in self.fields]
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.content_type = data.get("content_type", "application/json")
        
        # Fields
        self.fields = []
        for field_data in data.get("fields", []):
            field = ApiField("", "")
            field.from_dict(field_data)
            self.fields.append(field)
            
        return self

class ApiResponse:
    """Represents a response from an API endpoint."""
    
    def __init__(self, content_type="application/json", fields=None, status_code=200):
        self.content_type = content_type
        self.fields = fields or []  # List of ApiField objects
        self.status_code = status_code
        
    def add_field(self, field):
        """Add a field to this response."""
        self.fields.append(field)
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "content_type": self.content_type,
            "fields": [field.to_dict() for field in self.fields],
            "status_code": self.status_code
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.content_type = data.get("content_type", "application/json")
        self.status_code = data.get("status_code", 200)
        
        # Fields
        self.fields = []
        for field_data in data.get("fields", []):
            field = ApiField("", "")
            field.from_dict(field_data)
            self.fields.append(field)
            
        return self

class ApiField:
    """Represents a field in a request body or response."""
    
    def __init__(self, name, data_type, description="", required=False, is_array=False, nested_fields=None):
        self.name = name
        self.data_type = data_type
        self.description = description
        self.required = required
        self.is_array = is_array
        self.nested_fields = nested_fields or []  # List of ApiField objects for nested fields
        
    def add_nested_field(self, field):
        """Add a nested field to this field."""
        self.nested_fields.append(field)
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description,
            "required": self.required,
            "is_array": self.is_array,
            "nested_fields": [field.to_dict() for field in self.nested_fields]
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.name = data.get("name", "")
        self.data_type = data.get("data_type", "")
        self.description = data.get("description", "")
        self.required = data.get("required", False)
        self.is_array = data.get("is_array", False)
        
        # Nested fields
        self.nested_fields = []
        for field_data in data.get("nested_fields", []):
            field = ApiField("", "")
            field.from_dict(field_data)
            self.nested_fields.append(field)
            
        return self

class ViewToApiMapping:
    """Represents a mapping between a database view and an API endpoint."""
    
    def __init__(self, view_name, endpoint_path, confidence=0.0, notes=""):
        self.view_name = view_name
        self.endpoint_path = endpoint_path
        self.confidence = confidence  # 0.0 to 1.0
        self.notes = notes
        self.field_mappings = []  # List of FieldMapping objects
        
    def add_field_mapping(self, field_mapping):
        """Add a field mapping to this mapping."""
        self.field_mappings.append(field_mapping)
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "view_name": self.view_name,
            "endpoint_path": self.endpoint_path,
            "confidence": self.confidence,
            "notes": self.notes,
            "field_mappings": [fm.to_dict() for fm in self.field_mappings]
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.view_name = data.get("view_name", "")
        self.endpoint_path = data.get("endpoint_path", "")
        self.confidence = data.get("confidence", 0.0)
        self.notes = data.get("notes", "")
        
        # Field mappings
        self.field_mappings = []
        for fm_data in data.get("field_mappings", []):
            fm = FieldMapping("", "")
            fm.from_dict(fm_data)
            self.field_mappings.append(fm)
            
        return self

class FieldMapping:
    """Represents a mapping between a view column and an API field."""
    
    def __init__(self, view_column, api_field, confidence=0.0, transformation="", notes=""):
        self.view_column = view_column
        self.api_field = api_field
        self.confidence = confidence  # 0.0 to 1.0
        self.transformation = transformation  # e.g., "uppercase", "concatenate with foo"
        self.notes = notes
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "view_column": self.view_column,
            "api_field": self.api_field,
            "confidence": self.confidence,
            "transformation": self.transformation,
            "notes": self.notes
        }
    
    def from_dict(self, data):
        """Populate from dictionary."""
        self.view_column = data.get("view_column", "")
        self.api_field = data.get("api_field", "")
        self.confidence = data.get("confidence", 0.0)
        self.transformation = data.get("transformation", "")
        self.notes = data.get("notes", "")
        return self


class ViewExtractor:
    """Extract database view definitions from Confluence content."""
    
    def __init__(self, confluence_client):
        """Initialize with Confluence client."""
        self.confluence = confluence_client
        self.views = {}  # Dictionary of views by name
        
    def extract_views_from_content(self, content, page=None):
        """
        Extract view definitions from content.
        
        Args:
            content: The HTML or text content to extract from
            page: The page object for source tracking
        
        Returns:
            List of DatabaseView objects extracted
        """
        views = []
        
        # Extract views from tables
        tables = self._extract_tables_from_content(content)
        for table in tables:
            table_views = self._extract_views_from_table(table, page)
            views.extend(table_views)
        
        # Extract views from text
        text_views = self._extract_views_from_text(content, page)
        views.extend(text_views)
        
        # Update the views dictionary
        for view in views:
            self.views[view.name] = view
        
        return views
    
    def _extract_tables_from_content(self, content):
        """Extract tables from content."""
        tables = []
        
        # Parse with BeautifulSoup if it's HTML
        if "<table" in content:
            soup = BeautifulSoup(content, 'html.parser')
            for table in soup.find_all('table'):
                tables.append(table)
        
        # Also extract tables in Markdown format
        # Find tables with | delimiter
        md_tables = re.findall(r'(\|.*\|[\r\n]+\|[\s-:]*\|[\r\n]+((?:\|.*\|[\r\n]+)+))', content)
        for md_table in md_tables:
            tables.append(md_table[0])
        
        return tables
    
    def _extract_views_from_table(self, table, page=None):
        """Extract view definitions from a table."""
        views = []
        
        # Process HTML table
        if isinstance(table, BeautifulSoup) or hasattr(table, 'find_all'):
            # Check if it looks like a table of views
            headers = []
            header_row = table.find('tr')
            if header_row:
                headers = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
            
            # Look for view-related headers
            view_table = any(header in headers for header in ['view', 'view name', 'database view'])
            
            if view_table:
                # Extract view info from each row
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        view_name = cells[headers.index('view') if 'view' in headers 
                                   else headers.index('view name') if 'view name' in headers
                                   else headers.index('database view')].text.strip()
                        
                        description = ""
                        if 'description' in headers:
                            description = cells[headers.index('description')].text.strip()
                        
                        view = DatabaseView(view_name, description)
                        view.source_page = page
                        
                        # Extract columns if available
                        if 'columns' in headers:
                            columns_text = cells[headers.index('columns')].text.strip()
                            columns = self._parse_columns_text(columns_text)
                            for col in columns:
                                view.add_column(col)
                        
                        views.append(view)
        
        # Process Markdown table
        elif isinstance(table, str):
            lines = table.strip().split('\n')
            if len(lines) >= 3:  # Header, separator, and at least one data row
                # Extract headers
                headers = [h.strip().lower() for h in lines[0].strip('|').split('|')]
                
                # Check if it's a view table
                view_table = any(header in headers for header in ['view', 'view name', 'database view'])
                
                if view_table:
                    # Extract view info from each row
                    for line in lines[2:]:  # Skip header and separator rows
                        cells = [cell.strip() for cell in line.strip('|').split('|')]
                        if len(cells) >= len(headers):
                            view_name_idx = next((i for i, h in enumerate(headers) 
                                             if h in ['view', 'view name', 'database view']), 0)
                            view_name = cells[view_name_idx]
                            
                            description = ""
                            if 'description' in headers:
                                description = cells[headers.index('description')]
                            
                            view = DatabaseView(view_name, description)
                            view.source_page = page
                            
                            # Extract columns if available
                            if 'columns' in headers:
                                columns_text = cells[headers.index('columns')]
                                columns = self._parse_columns_text(columns_text)
                                for col in columns:
                                    view.add_column(col)
                            
                            views.append(view)
        
        return views
    
    def _extract_views_from_text(self, content, page=None):
        """Extract view definitions from text content using regex patterns."""
        views = []
        
        # Look for view definitions in text
        view_patterns = [
            # Pattern for "CREATE VIEW view_name AS"
            r'CREATE\s+VIEW\s+([a-zA-Z0-9_]+)(?:\s+AS\s+)?',
            # Pattern for "View: view_name"
            r'View:\s+([a-zA-Z0-9_]+)',
            # Pattern for "Database View: view_name"
            r'Database\s+View:\s+([a-zA-Z0-9_]+)',
            # Pattern for "## view_name (View)"
            r'##\s+([a-zA-Z0-9_]+)\s+\(View\)'
        ]
        
        # Find all potential view names
        view_names = set()
        for pattern in view_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                view_name = match.group(1).strip()
                view_names.add(view_name)
        
        # For each potential view, try to extract its definition
        for view_name in view_names:
            view = DatabaseView(view_name)
            view.source_page = page
            
            # Look for description
            desc_pattern = r'{}[\s\n]*-[\s\n]*(.*?)(?:[\r\n]{{2,}}|$)'.format(re.escape(view_name))
            desc_match = re.search(desc_pattern, content, re.IGNORECASE | re.DOTALL)
            if desc_match:
                view.description = desc_match.group(1).strip()
            
            # Look for columns
            col_patterns = [
                # Pattern for table-like listing
                r'{}.*?Columns:[\s\n]+(.*?)(?:[\r\n]{{2,}}|$)'.format(re.escape(view_name)),
                # Pattern for column list after view name
                r'{}.*?columns:[\s\n]+(.*?)(?:[\r\n]{{2,}}|$)'.format(re.escape(view_name))
            ]
            
            for pattern in col_patterns:
                col_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if col_match:
                    columns_text = col_match.group(1).strip()
                    columns = self._parse_columns_text(columns_text)
                    for col in columns:
                        view.add_column(col)
                    break
            
            views.append(view)
        
        return views
    
    def _parse_columns_text(self, columns_text):
        """Parse a text description of columns into ViewColumn objects."""
        columns = []
        
        # Split by commas or newlines
        if ',' in columns_text:
            col_parts = columns_text.split(',')
        else:
            col_parts = columns_text.split('\n')
        
        for part in col_parts:
            part = part.strip()
            if not part:
                continue
            
            # Try different parsing patterns
            
            # Pattern: "column_name (data_type) - description"
            match = re.match(r'([a-zA-Z0-9_]+)\s*\(([^)]+)\)\s*-\s*(.*)', part)
            if match:
                name, data_type, description = match.groups()
                columns.append(ViewColumn(name.strip(), data_type.strip(), description.strip()))
                continue
            
            # Pattern: "column_name (data_type)"
            match = re.match(r'([a-zA-Z0-9_]+)\s*\(([^)]+)\)', part)
            if match:
                name, data_type = match.groups()
                columns.append(ViewColumn(name.strip(), data_type.strip()))
                continue
            
            # Pattern: "column_name: data_type"
            match = re.match(r'([a-zA-Z0-9_]+):\s*(.*)', part)
            if match:
                name, data_type = match.groups()
                columns.append(ViewColumn(name.strip(), data_type.strip()))
                continue
            
            # Fallback: assume it's just a column name
            columns.append(ViewColumn(part, ""))
        
        return columns
    
    def extract_views_from_all_pages(self, pages):
        """
        Extract views from all pages.
        
        Args:
            pages: List of page objects with content
            
        Returns:
            Dictionary of views by name
        """
        logger.info(f"Extracting views from {len(pages)} pages")
        
        for page in pages:
            if "content" in page:
                self.extract_views_from_content(page["content"], page)
        
        logger.info(f"Extracted {len(self.views)} views")
        return self.views

class ApiExtractor:
    """Extract API endpoint definitions from Confluence content."""
    
    def __init__(self, confluence_client):
        """Initialize with Confluence client."""
        self.confluence = confluence_client
        self.endpoints = {}  # Dictionary of endpoints by path+method
        
    def extract_endpoints_from_content(self, content, page=None):
        """
        Extract API endpoint definitions from content.
        
        Args:
            content: The HTML or text content to extract from
            page: The page object for source tracking
            
        Returns:
            List of ApiEndpoint objects extracted
        """
        endpoints = []
        
        # Extract endpoints from tables
        tables = self._extract_tables_from_content(content)
        for table in tables:
            table_endpoints = self._extract_endpoints_from_table(table, page)
            endpoints.extend(table_endpoints)
        
        # Extract endpoints from text
        text_endpoints = self._extract_endpoints_from_text(content, page)
        endpoints.extend(text_endpoints)
        
        # Update the endpoints dictionary
        for endpoint in endpoints:
            key = f"{endpoint.method}:{endpoint.path}"
            self.endpoints[key] = endpoint
        
        return endpoints
    
    def _extract_tables_from_content(self, content):
        """Extract tables from content."""
        tables = []
        
        # Parse with BeautifulSoup if it's HTML
        if "<table" in content:
            soup = BeautifulSoup(content, 'html.parser')
            for table in soup.find_all('table'):
                tables.append(table)
        
        # Also extract tables in Markdown format
        # Find tables with | delimiter
        md_tables = re.findall(r'(\|.*\|[\r\n]+\|[\s-:]*\|[\r\n]+((?:\|.*\|[\r\n]+)+))', content)
        for md_table in md_tables:
            tables.append(md_table[0])
        
        return tables
    
    def _extract_endpoints_from_table(self, table, page=None):
        """Extract API endpoint definitions from a table."""
        endpoints = []
        
        # Process HTML table
        if isinstance(table, BeautifulSoup) or hasattr(table, 'find_all'):
            # Check if it looks like a table of API endpoints
            headers = []
            header_row = table.find('tr')
            if header_row:
                headers = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
            
            # Look for endpoint-related headers
            api_table = any(header in headers for header in ['endpoint', 'path', 'api', 'url', 'uri'])
            
            if api_table:
                # Extract endpoint info from each row
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        # Find the endpoint path
                        path_idx = next((i for i, h in enumerate(headers) 
                                        if h in ['endpoint', 'path', 'api', 'url', 'uri']), 0)
                        path = cells[path_idx].text.strip()
                        
                        # Find the HTTP method
                        method = "GET"  # Default
                        if 'method' in headers:
                            method = cells[headers.index('method')].text.strip().upper()
                        
                        # Find the description
                        description = ""
                        if 'description' in headers:
                            description = cells[headers.index('description')].text.strip()
                        
                        endpoint = ApiEndpoint(path, method, description)
                        endpoint.source_page = page
                        
                        # Extract parameters if available
                        if 'parameters' in headers:
                            param_text = cells[headers.index('parameters')].text.strip()
                            params = self._parse_parameters_text(param_text)
                            for param in params:
                                endpoint.add_parameter(param)
                        
                        # Extract request body if available
                        if 'request body' in headers or 'request' in headers:
                            idx = headers.index('request body') if 'request body' in headers else headers.index('request')
                            req_body_text = cells[idx].text.strip()
                            if req_body_text:
                                req_body = self._parse_request_body_text(req_body_text)
                                endpoint.set_request_body(req_body)
                        
                        # Extract response if available
                        if 'response' in headers:
                            resp_text = cells[headers.index('response')].text.strip()
                            if resp_text:
                                response = self._parse_response_text(resp_text)
                                endpoint.set_response(response)
                        
                        endpoints.append(endpoint)
        
        # Process Markdown table
        elif isinstance(table, str):
            lines = table.strip().split('\n')
            if len(lines) >= 3:  # Header, separator, and at least one data row
                # Extract headers
                headers = [h.strip().lower() for h in lines[0].strip('|').split('|')]
                
                # Check if it's an API table
                api_table = any(header in headers for header in ['endpoint', 'path', 'api', 'url', 'uri'])
                
                if api_table:
                    # Extract endpoint info from each row
                    for line in lines[2:]:  # Skip header and separator rows
                        cells = [cell.strip() for cell in line.strip('|').split('|')]
                        if len(cells) >= len(headers):
                            # Find the endpoint path
                            path_idx = next((i for i, h in enumerate(headers) 
                                          if h in ['endpoint', 'path', 'api', 'url', 'uri']), 0)
                            path = cells[path_idx]
                            
                            # Find the HTTP method
                            method = "GET"  # Default
                            if 'method' in headers:
                                method = cells[headers.index('method')].upper()
                            
                            # Find the description
                            description = ""
                            if 'description' in headers:
                                description = cells[headers.index('description')]
                            
                            endpoint = ApiEndpoint(path, method, description)
                            endpoint.source_page = page
                            
                            # Extract parameters if available
                            if 'parameters' in headers:
                                param_text = cells[headers.index('parameters')]
                                params = self._parse_parameters_text(param_text)
                                for param in params:
                                    endpoint.add_parameter(param)
                            
                            # Extract request body if available
                            if 'request body' in headers or 'request' in headers:
                                idx = headers.index('request body') if 'request body' in headers else headers.index('request')
                                req_body_text = cells[idx]
                                if req_body_text:
                                    req_body = self._parse_request_body_text(req_body_text)
                                    endpoint.set_request_body(req_body)
                            
                            # Extract response if available
                            if 'response' in headers:
                                resp_text = cells[headers.index('response')]
                                if resp_text:
                                    response = self._parse_response_text(resp_text)
                                    endpoint.set_response(response)
                            
                            endpoints.append(endpoint)
        
        return endpoints
    
    def _extract_endpoints_from_text(self, content, page=None):
        """Extract API endpoint definitions from text content using regex patterns."""
        endpoints = []
        
        # Look for API endpoint definitions in text
        endpoint_patterns = [
            # Pattern for "Endpoint: /path/to/api"
            r'Endpoint:\s+(\/[a-zA-Z0-9_\-\/{}]+)',
            # Pattern for "Path: /path/to/api"
            r'Path:\s+(\/[a-zA-Z0-9_\-\/{}]+)',
            # Pattern for "## /path/to/api"
            r'##\s+(\/[a-zA-Z0-9_\-\/{}]+)',
            # Pattern for code blocks with paths
            r'```[^\n]*\n(?:[^\n]*\n)*([A-Z]+)\s+(\/[a-zA-Z0-9_\-\/{}]+)',
        ]
        
        # Find all potential endpoint paths
        for pattern in endpoint_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match.groups()) == 1:
                    path = match.group(1).strip()
                    method = "GET"  # Default
                else:
                    method = match.group(1).strip()
                    path = match.group(2).strip()
                
                # Check if we already have this endpoint
                key = f"{method}:{path}"
                if key in self.endpoints:
                    continue
                
                # Try to find description
                desc_pattern = r'{}.*?Description:[\s\n]+(.*?)(?:[\r\n]{{2,}}|Parameters:|Request:|Response:|$)'.format(
                    re.escape(path))
                desc_match = re.search(desc_pattern, content, re.IGNORECASE | re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""
                
                endpoint = ApiEndpoint(path, method, description)
                endpoint.source_page = page
                
                # Look for parameters
                param_pattern = r'{}.*?Parameters:[\s\n]+(.*?)(?:[\r\n]{{2,}}|Request:|Response:|$)'.format(
                    re.escape(path))
                param_match = re.search(param_pattern, content, re.IGNORECASE | re.DOTALL)
                if param_match:
                    param_text = param_match.group(1).strip()
                    params = self._parse_parameters_text(param_text)
                    for param in params:
                        endpoint.add_parameter(param)
                
                # Look for request body
                req_pattern = r'{}.*?Request:[\s\n]+(.*?)(?:[\r\n]{{2,}}|Response:|$)'.format(
                    re.escape(path))
                req_match = re.search(req_pattern, content, re.IGNORECASE | re.DOTALL)
                if req_match:
                    req_body_text = req_match.group(1).strip()
                    req_body = self._parse_request_body_text(req_body_text)
                    endpoint.set_request_body(req_body)
                
                # Look for response
                resp_pattern = r'{}.*?Response:[\s\n]+(.*?)(?:[\r\n]{{2,}}|$)'.format(
                    re.escape(path))
                resp_match = re.search(resp_pattern, content, re.IGNORECASE | re.DOTALL)
                if resp_match:
                    resp_text = resp_match.group(1).strip()
                    response = self._parse_response_text(resp_text)
                    endpoint.set_response(response)
                
                endpoints.append(endpoint)
        
        return endpoints
    
    def _parse_parameters_text(self, param_text):
        """Parse a text description of parameters into ApiParameter objects."""
        parameters = []
        
        # Split by commas or newlines
        if ',' in param_text and '\n' not in param_text:
            param_parts = param_text.split(',')
        else:
            param_parts = param_text.split('\n')
        
        for part in param_parts:
            part = part.strip()
            if not part:
                continue
            
            # Try different parsing patterns
            
            # Pattern: "name (type) - description [required]"
            match = re.match(r'([a-zA-Z0-9_]+)\s*\(([^)]+)\)\s*-\s*(.*?)(?:\s*\[([^]]+)\])?$', part)
            if match:
                name, param_type, description, required = match.groups() + (None,) * (4 - len(match.groups()))
                required_bool = required and required.lower() == 'required'
                parameters.append(ApiParameter(name.strip(), param_type.strip(), description.strip(), required_bool))
                continue
            
            # Pattern: "name: type - description"
            match = re.match(r'([a-zA-Z0-9_]+):\s*([^-]+)\s*-\s*(.*)', part)
            if match:
                name, param_type, description = match.groups()
                # Check for [required] in description
                required_bool = '[required]' in description.lower()
                if required_bool:
                    description = description.replace('[required]', '').replace('[Required]', '').strip()
                parameters.append(ApiParameter(name.strip(), param_type.strip(), description.strip(), required_bool))
                continue
            
            # Pattern: "name: type"
            match = re.match(r'([a-zA-Z0-9_]+):\s*(.*)', part)
            if match:
                name, param_type = match.groups()
                parameters.append(ApiParameter(name.strip(), param_type.strip()))
                continue
            
            # Fallback: assume it's just a parameter name
            parameters.append(ApiParameter(part, ""))
        
        return parameters
    
    def _parse_request_body_text(self, req_body_text):
        """Parse a text description of a request body into an ApiRequestBody object."""
        req_body = ApiRequestBody()
        
        # Try to determine content type
        if 'application/json' in req_body_text.lower():
            req_body.content_type = 'application/json'
        elif 'application/xml' in req_body_text.lower():
            req_body.content_type = 'application/xml'
        elif 'multipart/form-data' in req_body_text.lower():
            req_body.content_type = 'multipart/form-data'
        
        # Extract fields from JSON-like structures
        # Look for JSON example
        json_match = re.search(r'```(?:json)?\s*\{(.*?)\}```', req_body_text, re.DOTALL)
        if json_match:
            json_text = '{' + json_match.group(1) + '}'
            try:
                # Try to parse as JSON
                json_data = json.loads(json_text)
                self._extract_fields_from_json(json_data, req_body)
            except json.JSONDecodeError:
                # If not valid JSON, try to parse fields manually
                field_matches = re.finditer(r'"([^"]+)"\s*:\s*([^,\n]+),?', json_text)
                for match in field_matches:
                    name = match.group(1)
                    value = match.group(2).strip()
                    data_type = self._infer_data_type(value)
                    field = ApiField(name, data_type)
                    req_body.add_field(field)
        else:
            # Try to extract fields from text description
            field_matches = re.finditer(r'([a-zA-Z0-9_]+)\s*\(([^)]+)\)(?:\s*-\s*(.*?))?(?=\n[a-zA-Z0-9_]+\s*\(|$)', 
                                       req_body_text, re.DOTALL)
            for match in field_matches:
                name, data_type, description = match.groups() + (None,) * (3 - len(match.groups()))
                description = description.strip() if description else ""
                required = '[required]' in description.lower()
                if required:
                    description = description.replace('[required]', '').replace('[Required]', '').strip()
                is_array = '[]' in data_type or 'array' in data_type.lower()
                field = ApiField(name.strip(), data_type.strip(), description, required, is_array)
                req_body.add_field(field)
        
        return req_body
    
    def _parse_response_text(self, resp_text):
        """Parse a text description of a response into an ApiResponse object."""
        response = ApiResponse()
        
        # Try to determine content type
        if 'application/json' in resp_text.lower():
            response.content_type = 'application/json'
        elif 'application/xml' in resp_text.lower():
            response.content_type = 'application/xml'
        
        # Try to determine status code
        status_match = re.search(r'(\d{3})\s*[:-]', resp_text)
        if status_match:
            response.status_code = int(status_match.group(1))
        
        # Extract fields from JSON-like structures
        # Look for JSON example
        json_match = re.search(r'```(?:json)?\s*\{(.*?)\}```', resp_text, re.DOTALL)
        if json_match:
            json_text = '{' + json_match.group(1) + '}'
            try:
                # Try to parse as JSON
                json_data = json.loads(json_text)
                self._extract_fields_from_json(json_data, response)
            except json.JSONDecodeError:
                # If not valid JSON, try to parse fields manually
                field_matches = re.finditer(r'"([^"]+)"\s*:\s*([^,\n]+),?', json_text)
                for match in field_matches:
                    name = match.group(1)
                    value = match.group(2).strip()
                    data_type = self._infer_data_type(value)
                    field = ApiField(name, data_type)
                    response.add_field(field)
        else:
            # Try to extract fields from text description
            field_matches = re.finditer(r'([a-zA-Z0-9_]+)\s*\(([^)]+)\)(?:\s*-\s*(.*?))?(?=\n[a-zA-Z0-9_]+\s*\(|$)', 
                                       resp_text, re.DOTALL)
            for match in field_matches:
                name, data_type, description = match.groups() + (None,) * (3 - len(match.groups()))
                description = description.strip() if description else ""
                is_array = '[]' in data_type or 'array' in data_type.lower()
                field = ApiField(name.strip(), data_type.strip(), description, False, is_array)
                response.add_field(field)
        
        return response
    
    def _extract_fields_from_json(self, json_data, container):
        """
        Extract fields from a JSON object.
        
        Args:
            json_data: JSON data (dict or list)
            container: ApiRequestBody or ApiResponse to add fields to
        """
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                data_type = self._infer_data_type(value)
                is_array = isinstance(value, list)
                field = ApiField(key, data_type, "", False, is_array)
                
                # Handle nested objects/arrays
                if isinstance(value, dict):
                    self._extract_fields_from_json(value, field)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    self._extract_fields_from_json(value[0], field)
                
                container.add_field(field)
    
    def _infer_data_type(self, value):
        """Infer data type from a value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            if value.lower() == "true" or value.lower() == "false":
                return "boolean"
            elif value.isdigit():
                return "integer"
            elif re.match(r'^-?\d+\.\d+$', value):
                return "number"
            elif re.match(r'^\d{4}-\d{2}-\d{2}', value):
                return "date"
            elif re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', value):
                return "datetime"
            elif value.startswith('"') and value.endswith('"'):
                return "string"
            else:
                return "string"
        elif isinstance(value, list):
            if value:
                return f"array of {self._infer_data_type(value[0])}"
            else:
                return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
    
    def extract_endpoints_from_all_pages(self, pages):
        """
        Extract API endpoints from all pages.
        
        Args:
            pages: List of page objects with content
            
        Returns:
            Dictionary of endpoints by path+method
        """
        logger.info(f"Extracting API endpoints from {len(pages)} pages")
        
        for page in pages:
            if "content" in page:
                self.extract_endpoints_from_content(page["content"], page)
        
        logger.info(f"Extracted {len(self.endpoints)} API endpoints")
        return self.endpoints



class MappingEngine:
    """Engine to create mappings between database views and API endpoints."""
    
    def __init__(self, views, endpoints, gemini_client=None):
        """
        Initialize with views and endpoints.
        
        Args:
            views: Dictionary of DatabaseView objects by name
            endpoints: Dictionary of ApiEndpoint objects by path+method
            gemini_client: GeminiAssistant for semantic analysis
        """
        self.views = views
        self.endpoints = endpoints
        self.gemini = gemini_client
        self.mappings = []  # List of ViewToApiMapping objects
    
    def generate_mappings(self):
        """
        Generate mappings between views and endpoints.
        
        Returns:
            List of ViewToApiMapping objects
        """
        logger.info(f"Generating mappings for {len(self.views)} views and {len(self.endpoints)} endpoints")
        
        # Use different mapping strategies
        self._apply_naming_convention_strategy()
        self._apply_content_similarity_strategy()
        
        # If we have a Gemini client, use it for semantic analysis
        if self.gemini:
            self._apply_gemini_semantic_analysis()
        
        logger.info(f"Generated {len(self.mappings)} mappings")
        return self.mappings
    
    def _apply_naming_convention_strategy(self):
        """Apply naming convention strategy to create mappings."""
        logger.info("Applying naming convention strategy")
        
        # Common patterns:
        # 1. view_name -> /api/view_names
        # 2. vw_resource -> /api/resources
        # 3. v_entity -> /api/entities
        
        view_names = list(self.views.keys())
        endpoint_paths = [endpoint.path for endpoint in self.endpoints.values()]
        
        for view_name in view_names:
            view = self.views[view_name]
            
            # Normalize view name (remove prefixes, make plural)
            base_name = view_name.lower()
            if base_name.startswith('vw_'):
                base_name = base_name[3:]
            elif base_name.startswith('v_'):
                base_name = base_name[2:]
            
            # Try different endpoint path patterns
            patterns = [
                f"/api/{base_name}s",  # Pluralized
                f"/api/{base_name}",   # Direct
                f"/{base_name}s",      # Root pluralized
                f"/{base_name}"        # Root direct
            ]
            
            matched_paths = []
            for pattern in patterns:
                for path in endpoint_paths:
                    path_lower = path.lower()
                    # Check if the path matches the pattern
                    if path_lower == pattern or path_lower.startswith(f"{pattern}/"):
                        matched_paths.append(path)
            
            # For each matched path, create mappings to the corresponding endpoints
            for path in matched_paths:
                # Find endpoints for this path
                path_endpoints = [e for e in self.endpoints.values() if e.path.lower() == path.lower()]
                
                for endpoint in path_endpoints:
                    # Calculate confidence based on method and path
                    confidence = 0.5  # Base confidence for name match
                    
                    # Higher confidence for expected REST patterns
                    if endpoint.method == "GET" and not '/' in path[path.rfind('/')+1:]:
                        # GET collection
                        confidence += 0.2
                    elif endpoint.method == "GET" and '{' in path:
                        # GET single item
                        confidence += 0.2
                    elif endpoint.method == "POST" and not '{' in path:
                        # POST to create
                        confidence += 0.2
                    elif endpoint.method == "PUT" and '{' in path:
                        # PUT to update
                        confidence += 0.2
                    elif endpoint.method == "DELETE" and '{' in path:
                        # DELETE single item
                        confidence += 0.2
                    
                    mapping = ViewToApiMapping(view_name, endpoint.path, confidence,
                                              f"Matched via naming convention ({view_name} -> {endpoint.path})")
                    
                    # Try to map fields
                    self._map_fields(view, endpoint, mapping)
                    
                    # Add the mapping
                    self.mappings.append(mapping)
    
    def _apply_content_similarity_strategy(self):
        """Apply content similarity strategy to create mappings."""
        logger.info("Applying content similarity strategy")
        
        # Compare field names between views and API endpoints
        for view_name, view in self.views.items():
            view_col_names = [col.name.lower() for col in view.columns]
            
            for endpoint_key, endpoint in self.endpoints.items():
                # Skip endpoints that don't have request or response fields
                if not (endpoint.request_body and endpoint.request_body.fields) and \
                   not (endpoint.response and endpoint.response.fields):
                    continue
                
                # Collect endpoint field names
                endpoint_fields = []
                if endpoint.request_body and endpoint.request_body.fields:
                    endpoint_fields.extend([f.name.lower() for f in endpoint.request_body.fields])
                if endpoint.response and endpoint.response.fields:
                    endpoint_fields.extend([f.name.lower() for f in endpoint.response.fields])
                
                # Calculate similarity based on field name overlap
                common_fields = set(view_col_names).intersection(set(endpoint_fields))
                if common_fields:
                    similarity = len(common_fields) / max(len(view_col_names), len(endpoint_fields))
                    
                    # Only create mapping if similarity is above threshold
                    if similarity > 0.3:
                        mapping = ViewToApiMapping(view_name, endpoint.path, similarity,
                                                  f"Matched via field similarity ({len(common_fields)} common fields)")
                        
                        # Try to map fields
                        self._map_fields(view, endpoint, mapping)
                        
                        # Add the mapping
                        self.mappings.append(mapping)
    
    def _apply_gemini_semantic_analysis(self):
        """Apply Gemini semantic analysis to create and refine mappings."""
        logger.info("Applying Gemini semantic analysis")
        
        # Collect unmapped views and endpoints
        mapped_views = set(mapping.view_name for mapping in self.mappings)
        mapped_endpoints = set(mapping.endpoint_path for mapping in self.mappings)
        
        unmapped_views = [v for k, v in self.views.items() if k not in mapped_views]
        unmapped_endpoints = [e for k, e in self.endpoints.items() 
                             if e.path not in mapped_endpoints]
        
        if not unmapped_views or not unmapped_endpoints:
            return
        
        # Get semantic analysis from Gemini
        mappings = self._get_gemini_mappings(unmapped_views, unmapped_endpoints)
        
        # Add the mappings
        for mapping in mappings:
            if mapping.view_name in self.views and \
               any(e.path == mapping.endpoint_path for e in self.endpoints.values()):
                view = self.views[mapping.view_name]
                endpoint = next(e for e in self.endpoints.values() if e.path == mapping.endpoint_path)
                
                # Try to map fields
                self._map_fields(view, endpoint, mapping)
                
                # Add the mapping
                self.mappings.append(mapping)
    
    def _get_gemini_mappings(self, views, endpoints):
        """
        Get mappings from Gemini.
        
        Args:
            views: List of unmapped DatabaseView objects
            endpoints: List of unmapped ApiEndpoint objects
            
        Returns:
            List of ViewToApiMapping objects
        """
        # Prepare prompt for Gemini
        prompt = self._create_mapping_prompt(views, endpoints)
        
        try:
            # Get response from Gemini
            response = self.gemini.generate_response(prompt)
            
            # Parse mappings from response
            mappings = self._parse_mappings_from_response(response, views, endpoints)
            return mappings
        except Exception as e:
            logger.error(f"Error getting mappings from Gemini: {str(e)}")
            return []
    
    def _create_mapping_prompt(self, views, endpoints):
        """Create a prompt for Gemini to analyze mappings."""
        prompt = "I need to map database views to REST API endpoints. Please analyze these views and endpoints and suggest mappings.\n\n"
        
        # Add view information
        prompt += "Database Views:\n"
        for view in views:
            prompt += f"- {view.name}: {view.description}\n"
            prompt += "  Columns:\n"
            for col in view.columns:
                prompt += f"  - {col.name} ({col.data_type}): {col.description}\n"
            prompt += "\n"
        
        # Add endpoint information
        prompt += "API Endpoints:\n"
        for endpoint in endpoints:
            prompt += f"- {endpoint.method} {endpoint.path}: {endpoint.description}\n"
            
            # Add parameters
            if endpoint.parameters:
                prompt += "  Parameters:\n"
                for param in endpoint.parameters:
                    prompt += f"  - {param.name} ({param.data_type}): {param.description}\n"
            
            # Add request body
            if endpoint.request_body and endpoint.request_body.fields:
                prompt += "  Request Body:\n"
                for field in endpoint.request_body.fields:
                    prompt += f"  - {field.name} ({field.data_type}): {field.description}\n"
            
            # Add response
            if endpoint.response and endpoint.response.fields:
                prompt += "  Response:\n"
                for field in endpoint.response.fields:
                    prompt += f"  - {field.name} ({field.data_type}): {field.description}\n"
            
            prompt += "\n"
        
        # Add instructions
        prompt += """
Please identify mappings between views and endpoints. For each mapping, provide:
1. The view name
2. The endpoint path
3. A confidence score (0.0 to 1.0)
4. Any notes explaining the mapping

Then for each mapping, provide field mappings between view columns and API fields.
Use this format for your response:

MAPPING 1:
View: <view_name>
Endpoint: <endpoint_path>
Confidence: <confidence>
Notes: <notes>
Field Mappings:
- <view_column> -> <api_field> [<transformation>]
- <view_column> -> <api_field> [<transformation>]

MAPPING 2:
...

Only include mappings that you believe are correct, with a confidence of at least 0.5.
"""
        
        return prompt
    
    def _parse_mappings_from_response(self, response, views, endpoints):
        """Parse mappings from Gemini response."""
        mappings = []
        
        # Split response into mapping sections
        mapping_sections = re.split(r'MAPPING \d+:', response)
        
        # Skip the first section (it's the intro text)
        for section in mapping_sections[1:]:
            section = section.strip()
            if not section:
                continue
            
            # Extract mapping details
            view_match = re.search(r'View:\s*(.+?)(?:\n|$)', section)
            endpoint_match = re.search(r'Endpoint:\s*(.+?)(?:\n|$)', section)
            confidence_match = re.search(r'Confidence:\s*(.+?)(?:\n|$)', section)
            notes_match = re.search(r'Notes:\s*(.+?)(?:\n|Field Mappings:|$)', section, re.DOTALL)
            
            if view_match and endpoint_match:
                view_name = view_match.group(1).strip()
                endpoint_path = endpoint_match.group(1).strip()
                confidence = float(confidence_match.group(1).strip()) if confidence_match else 0.5
                notes = notes_match.group(1).strip() if notes_match else ""
                
                mapping = ViewToApiMapping(view_name, endpoint_path, confidence, notes)
                
                # Extract field mappings
                field_mappings_match = re.search(r'Field Mappings:\s*\n(.*?)(?:\n\n|$)', section, re.DOTALL)
                if field_mappings_match:
                    field_mappings_text = field_mappings_match.group(1).strip()
                    field_mapping_lines = field_mappings_text.split('\n')
                    
                    for line in field_mapping_lines:
                        line = line.strip()
                        if not line or not line.startswith('-'):
                            continue
                        
                        # Extract field mapping details
                        field_mapping_match = re.match(r'-\s*(.+?)\s*->\s*(.+?)(?:\s*\[(.+?)\])?$', line)
                        if field_mapping_match:
                            view_col = field_mapping_match.group(1).strip()
                            api_field = field_mapping_match.group(2).strip()
                            transformation = field_mapping_match.group(3).strip() if field_mapping_match.group(3) else ""
                            
                            field_mapping = FieldMapping(view_col, api_field, confidence, transformation)
                            mapping.add_field_mapping(field_mapping)
                
                mappings.append(mapping)
        
        return mappings
    
    def _map_fields(self, view, endpoint, mapping):
        """
        Map fields between a view and an endpoint.
        
        Args:
            view: DatabaseView object
            endpoint: ApiEndpoint object
            mapping: ViewToApiMapping object to add field mappings to
        """
        # Collect view columns
        view_cols = {col.name.lower(): col for col in view.columns}
        
        # Collect endpoint fields
        endpoint_fields = {}
        
        # Request body fields
        if endpoint.request_body and endpoint.request_body.fields:
            for field in endpoint.request_body.fields:
                endpoint_fields[field.name.lower()] = field
        
        # Response fields
        if endpoint.response and endpoint.response.fields:
            for field in endpoint.response.fields:
                endpoint_fields[field.name.lower()] = field
        
        # Map fields with exact name matches
        for col_name, col in view_cols.items():
            if col_name in endpoint_fields:
                field_mapping = FieldMapping(col.name, endpoint_fields[col_name].name, 0.9)
                mapping.add_field_mapping(field_mapping)
        
        # Map fields with similar names
        for col_name, col in view_cols.items():
            if col_name in endpoint_fields:
                continue  # Already mapped
            
            # Check for similar field names
            for field_name, field in endpoint_fields.items():
                if field_name in [fm.api_field.lower() for fm in mapping.field_mappings]:
                    continue  # Field already mapped
                
                # Check for common patterns
                if field_name == f"{col_name}id" or col_name == f"{field_name}id":
                    field_mapping = FieldMapping(col.name, field.name, 0.7)
                    mapping.add_field_mapping(field_mapping)
                    break
                
                # Check for singular/plural variations
                if field_name + 's' == col_name or col_name + 's' == field_name:
                    field_mapping = FieldMapping(col.name, field.name, 0.7)
                    mapping.add_field_mapping(field_mapping)
                    break
                
                # Check for name without underscores/dashes
                col_name_clean = col_name.replace('_', '').replace('-', '')
                field_name_clean = field_name.replace('_', '').replace('-', '')
                if col_name_clean == field_name_clean:
                    field_mapping = FieldMapping(col.name, field.name, 0.8, "Remove special characters")
                    mapping.add_field_mapping(field_mapping)
                    break
    
    def get_mappings_for_view(self, view_name):
        """
        Get mappings for a specific view.
        
        Args:
            view_name: Name of the view
            
        Returns:
            List of ViewToApiMapping objects for the specified view
        """
        return [m for m in self.mappings if m.view_name.lower() == view_name.lower()]
    
    def get_mappings_for_endpoint(self, endpoint_path):
        """
        Get mappings for a specific endpoint.
        
        Args:
            endpoint_path: Path of the endpoint
            
        Returns:
            List of ViewToApiMapping objects for the specified endpoint
        """
        return [m for m in self.mappings if m.endpoint_path.lower() == endpoint_path.lower()]
    
    def save_mappings_to_file(self, filename):
        """
        Save mappings to a file.
        
        Args:
            filename: Name of the file to save to
        """
        with open(filename, 'w') as f:
            json.dump([m.to_dict() for m in self.mappings], f, indent=2)
        
        logger.info(f"Saved {len(self.mappings)} mappings to {filename}")
    
    def load_mappings_from_file(self, filename):
        """
        Load mappings from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            List of ViewToApiMapping objects
        """
        try:
            with open(filename, 'r') as f:
                mapping_dicts = json.load(f)
            
            self.mappings = []
            for mapping_dict in mapping_dicts:
                mapping = ViewToApiMapping("", "")
                mapping.from_dict(mapping_dict)
                self.mappings.append(mapping)
            
            logger.info(f"Loaded {len(self.mappings)} mappings from {filename}")
            return self.mappings
        except Exception as e:
            logger.error(f"Error loading mappings from {filename}: {str(e)}")
            return []



class CopperMapper:
    """Main class for the COPPER View-to-API Mapper system."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None):
        """
        Initialize the COPPER Mapper.
        
        Args:
            confluence_url: URL of the Confluence instance
            confluence_username: Username for Confluence
            confluence_api_token: API token for Confluence
            space_key: Key of the Confluence space to search in
        """
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.gemini = GeminiAssistant()
        self.space_key = space_key
        self.space_pages = []
        self.views = {}
        self.endpoints = {}
        self.mappings = []
        
        logger.info(f"Initialized COPPER Mapper for space: {space_key}")
    
    def initialize(self):
        """Initialize by loading all content and extracting views and endpoints."""
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and URL.")
            return False
        
        # Load all pages from the space
        self.load_space_content()
        
        # Extract views and endpoints
        self.extract_views_and_endpoints()
        
        # Generate mappings
        self.generate_mappings()
        
        return True
    
    def load_space_content(self):
        """Load all pages from the specified space."""
        logger.info(f"Loading all pages from space {self.space_key}")
        
        if not self.space_key:
            logger.error("No space key specified. Please provide a space key.")
            return
        
        # Get all pages in the space
        self.space_pages = self.confluence.get_all_pages_in_space(self.space_key)
        
        logger.info(f"Loaded metadata for {len(self.space_pages)} pages from space {self.space_key}")
        
        # Fetch content for all pages
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            page_ids = [page["id"] for page in self.space_pages]
            page_contents = list(executor.map(self.confluence.get_page_content, page_ids))
        
        # Add content to pages
        for i, page in enumerate(self.space_pages):
            if page_contents[i]:
                page["content"] = page_contents[i]["content"]
    
    def extract_views_and_endpoints(self):
        """Extract database views and API endpoints from the content."""
        logger.info("Extracting views and endpoints from content")
        
        # Extract views
        view_extractor = ViewExtractor(self.confluence)
        self.views = view_extractor.extract_views_from_all_pages(self.space_pages)
        
        # Extract endpoints
        api_extractor = ApiExtractor(self.confluence)
        self.endpoints = api_extractor.extract_endpoints_from_all_pages(self.space_pages)
        
        logger.info(f"Extracted {len(self.views)} views and {len(self.endpoints)} endpoints")
    
    def generate_mappings(self):
        """Generate mappings between views and endpoints."""
        logger.info("Generating mappings between views and endpoints")
        
        mapping_engine = MappingEngine(self.views, self.endpoints, self.gemini)
        self.mappings = mapping_engine.generate_mappings()
        
        logger.info(f"Generated {len(self.mappings)} mappings")
    
    def save_to_files(self, output_dir):
        """
        Save all extracted data to files.
        
        Args:
            output_dir: Directory to save files to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save views
        with open(os.path.join(output_dir, "views.json"), 'w') as f:
            json.dump({name: view.to_dict() for name, view in self.views.items()}, f, indent=2)
        
        # Save endpoints
        with open(os.path.join(output_dir, "endpoints.json"), 'w') as f:
            json.dump({key: endpoint.to_dict() for key, endpoint in self.endpoints.items()}, f, indent=2)
        
        # Save mappings
        with open(os.path.join(output_dir, "mappings.json"), 'w') as f:
            json.dump([mapping.to_dict() for mapping in self.mappings], f, indent=2)
        
        logger.info(f"Saved all data to {output_dir}")
    
    def load_from_files(self, input_dir):
        """
        Load all data from files.
        
        Args:
            input_dir: Directory to load files from
        """
        try:
            # Load views
            with open(os.path.join(input_dir, "views.json"), 'r') as f:
                view_dicts = json.load(f)
                self.views = {}
                for name, view_dict in view_dicts.items():
                    view = DatabaseView("")
                    view.from_dict(view_dict)
                    self.views[name] = view
            
            # Load endpoints
            with open(os.path.join(input_dir, "endpoints.json"), 'r') as f:
                endpoint_dicts = json.load(f)
                self.endpoints = {}
                for key, endpoint_dict in endpoint_dicts.items():
                    endpoint = ApiEndpoint("")
                    endpoint.from_dict(endpoint_dict)
                    self.endpoints[key] = endpoint
            
            # Load mappings
            with open(os.path.join(input_dir, "mappings.json"), 'r') as f:
                mapping_dicts = json.load(f)
                self.mappings = []
                for mapping_dict in mapping_dicts:
                    mapping = ViewToApiMapping("", "")
                    mapping.from_dict(mapping_dict)
                    self.mappings.append(mapping)
            
            logger.info(f"Loaded {len(self.views)} views, {len(self.endpoints)} endpoints, "
                      f"and {len(self.mappings)} mappings from {input_dir}")
            return True
        except Exception as e:
            logger.error(f"Error loading data from {input_dir}: {str(e)}")
            return False
    
    def query_mappings(self, query):
        """
        Query the mappings using natural language.
        
        Args:
            query: Natural language query about mappings
            
        Returns:
            Response to the query
        """
        # If we have no mappings, return an error
        if not self.mappings:
            return "No mappings available. Please extract views and endpoints and generate mappings first."
        
        # Create a context for Gemini with information about the mappings
        context = self._create_mapping_context()
        
        # Generate response using Gemini
        response = self.gemini.generate_response(query, context)
        
        return response
    
    def _create_mapping_context(self):
        """Create a context for Gemini with information about the mappings."""
        context = "Database Views:\n"
        for name, view in self.views.items():
            context += f"- {name}: {view.description}\n"
            context += "  Columns:\n"
            for col in view.columns:
                context += f"  - {col.name} ({col.data_type}): {col.description}\n"
            context += "\n"
        
        context += "API Endpoints:\n"
        for key, endpoint in self.endpoints.items():
            context += f"- {endpoint.method} {endpoint.path}: {endpoint.description}\n"
            
            # Add parameters
            if endpoint.parameters:
                context += "  Parameters:\n"
                for param in endpoint.parameters:
                    context += f"  - {param.name} ({param.data_type}): {param.description}\n"
            
            # Add request body
            if endpoint.request_body and endpoint.request_body.fields:
                context += "  Request Body:\n"
                for field in endpoint.request_body.fields:
                    context += f"  - {field.name} ({field.data_type}): {field.description}\n"
            
            # Add response
            if endpoint.response and endpoint.response.fields:
                context += "  Response:\n"
                for field in endpoint.response.fields:
                    context += f"  - {field.name} ({field.data_type}): {field.description}\n"
            
            context += "\n"
        
        context += "View-to-API Mappings:\n"
        for mapping in self.mappings:
            context += f"- View: {mapping.view_name} -> Endpoint: {mapping.endpoint_path} (Confidence: {mapping.confidence:.2f})\n"
            context += f"  Notes: {mapping.notes}\n"
            
            # Add field mappings
            if mapping.field_mappings:
                context += "  Field Mappings:\n"
                for fm in mapping.field_mappings:
                    transform_text = f" [{fm.transformation}]" if fm.transformation else ""
                    context += f"  - {fm.view_column} -> {fm.api_field}{transform_text}\n"
            
            context += "\n"
        
        return context
    
    def generate_api_code(self, view_name, language="python"):
        """
        Generate sample code to access the API for a given view.
        
        Args:
            view_name: Name of the view to generate code for
            language: Programming language to generate code in
            
        Returns:
            Generated code
        """
        # Find mappings for the view
        view_mappings = [m for m in self.mappings if m.view_name.lower() == view_name.lower()]
        
        if not view_mappings:
            return f"No mappings found for view: {view_name}"
        
        # Sort mappings by confidence
        view_mappings.sort(key=lambda m: m.confidence, reverse=True)
        
        # Get the highest confidence mapping
        mapping = view_mappings[0]
        
        # Find the endpoint
        endpoint = None
        for ep in self.endpoints.values():
            if ep.path.lower() == mapping.endpoint_path.lower():
                endpoint = ep
                break
        
        if not endpoint:
            return f"Endpoint not found for mapping: {mapping.endpoint_path}"
        
        # Create a prompt for Gemini
        prompt = f"""Generate sample {language} code to access the API endpoint that corresponds to the database view "{view_name}".

View: {view_name}
API Endpoint: {endpoint.method} {endpoint.path} ({endpoint.description})

Parameters:
"""
        for param in endpoint.parameters:
            prompt += f"- {param.name} ({param.data_type}): {param.description}\n"
        
        prompt += "\nRequest Body:\n"
        if endpoint.request_body and endpoint.request_body.fields:
            for field in endpoint.request_body.fields:
                prompt += f"- {field.name} ({field.data_type}): {field.description}\n"
        else:
            prompt += "N/A\n"
        
        prompt += "\nResponse:\n"
        if endpoint.response and endpoint.response.fields:
            for field in endpoint.response.fields:
                prompt += f"- {field.name} ({field.data_type}): {field.description}\n"
        else:
            prompt += "N/A\n"
        
        prompt += "\nField Mappings:\n"
        for fm in mapping.field_mappings:
            transform_text = f" [{fm.transformation}]" if fm.transformation else ""
            prompt += f"- {fm.view_column} -> {fm.api_field}{transform_text}\n"
        
        prompt += f"\nPlease generate sample {language} code to access this API endpoint, with proper error handling and processing of the response."
        
        # Generate code using Gemini
        code = self.gemini.generate_response(prompt)
        
        return code
    
    def generate_documentation(self, output_file):
        """
        Generate comprehensive documentation of all mappings.
        
        Args:
            output_file: File to save documentation to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                f.write("# COPPER View-to-API Mapping Documentation\n\n")
                
                # Overall statistics
                f.write("## Overview\n\n")
                f.write(f"- Total Views: {len(self.views)}\n")
                f.write(f"- Total API Endpoints: {len(self.endpoints)}\n")
                f.write(f"- Total Mappings: {len(self.mappings)}\n\n")
                
                # Views
                f.write("## Database Views\n\n")
                for name, view in sorted(self.views.items()):
                    f.write(f"### {name}\n\n")
                    f.write(f"{view.description}\n\n")
                    
                    f.write("**Columns:**\n\n")
                    f.write("| Column | Type | Description |\n")
                    f.write("|--------|------|-------------|\n")
                    for col in view.columns:
                        f.write(f"| {col.name} | {col.data_type} | {col.description} |\n")
                    
                    f.write("\n**Primary Keys:** ")
                    if view.primary_keys:
                        f.write(", ".join(view.primary_keys))
                    else:
                        f.write("None specified")
                    
                    f.write("\n\n**Mapped Endpoints:**\n\n")
                    view_mappings = [m for m in self.mappings if m.view_name.lower() == name.lower()]
                    if view_mappings:
                        for mapping in sorted(view_mappings, key=lambda m: m.confidence, reverse=True):
                            f.write(f"- [{mapping.endpoint_path}](#{mapping.endpoint_path.replace('/', '-')}) "
                                   f"(Confidence: {mapping.confidence:.2f})\n")
                    else:
                        f.write("No mappings found\n")
                    
                    f.write("\n")
                
                # API Endpoints
                f.write("## API Endpoints\n\n")
                for key, endpoint in sorted(self.endpoints.items(), key=lambda x: x[1].path):
                    anchor = endpoint.path.replace('/', '-')
                    f.write(f"### <a name=\"{anchor}\"></a>{endpoint.method} {endpoint.path}\n\n")
                    f.write(f"{endpoint.description}\n\n")
                    
                    if endpoint.parameters:
                        f.write("**Parameters:**\n\n")
                        f.write("| Parameter | Type | Description | Required | Location |\n")
                        f.write("|-----------|------|-------------|----------|----------|\n")
                        for param in endpoint.parameters:
                            f.write(f"| {param.name} | {param.data_type} | {param.description} | "
                                   f"{'Yes' if param.required else 'No'} | {param.location} |\n")
                        f.write("\n")
                    
                    if endpoint.request_body and endpoint.request_body.fields:
                        f.write("**Request Body:** ")
                        f.write(f"{endpoint.request_body.content_type}\n\n")
                        f.write("| Field | Type | Description | Required |\n")
                        f.write("|-------|------|-------------|----------|\n")
                        for field in endpoint.request_body.fields:
                            f.write(f"| {field.name} | {field.data_type} | {field.description} | "
                                   f"{'Yes' if field.required else 'No'} |\n")
                        f.write("\n")
                    
                    if endpoint.response and endpoint.response.fields:
                        f.write("**Response:** ")
                        f.write(f"{endpoint.response.content_type} (Status: {endpoint.response.status_code})\n\n")
                        f.write("| Field | Type | Description |\n")
                        f.write("|-------|------|-------------|\n")
                        for field in endpoint.response.fields:
                            f.write(f"| {field.name} | {field.data_type} | {field.description} |\n")
                        f.write("\n")
                    
                    f.write("**Mapped Views:**\n\n")
                    endpoint_mappings = [m for m in self.mappings if m.endpoint_path.lower() == endpoint.path.lower()]
                    if endpoint_mappings:
                        for mapping in sorted(endpoint_mappings, key=lambda m: m.confidence, reverse=True):
                            f.write(f"- [{mapping.view_name}](#{mapping.view_name}) "
                                   f"(Confidence: {mapping.confidence:.2f})\n")
                    else:
                        f.write("No mappings found\n")
                    
                    f.write("\n")
                
                # Mappings
                f.write("## View-to-API Mappings\n\n")
                for mapping in sorted(self.mappings, key=lambda m: (m.view_name, m.confidence), reverse=True):
                    f.write(f"### {mapping.view_name} -> {mapping.endpoint_path}\n\n")
                    f.write(f"**Confidence:** {mapping.confidence:.2f}\n\n")
                    f.write(f"**Notes:** {mapping.notes}\n\n")
                    
                    f.write("**Field Mappings:**\n\n")
                    if mapping.field_mappings:
                        f.write("| View Column | API Field | Confidence | Transformation |\n")
                        f.write("|-------------|-----------|------------|----------------|\n")
                        for fm in mapping.field_mappings:
                            f.write(f"| {fm.view_column} | {fm.api_field} | {fm.confidence:.2f} | {fm.transformation} |\n")
                    else:
                        f.write("No field mappings found\n")
                    
                    # Generate code examples
                    f.write("\n**Code Example:**\n\n")
                    f.write("```python\n")
                    code = self.generate_api_code(mapping.view_name, "python")
                    f.write(code)
                    f.write("\n```\n\n")
            
            logger.info(f"Generated documentation saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            return False


def main():
    """Main entry point for the COPPER Mapper."""
    logger.info("Starting COPPER Mapper")
    
    # Check for required environment variables
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN or not CONFLUENCE_URL:
        logger.error("Missing Confluence credentials. Please set CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, and CONFLUENCE_URL environment variables.")
        print("Error: Missing Confluence credentials. Please set the required environment variables.")
        return
    
    parser = argparse.ArgumentParser(description="COPPER View-to-API Mapper")
    parser.add_argument("--space", help="Confluence space key", default=CONFLUENCE_SPACE)
    parser.add_argument("--output", help="Output directory", default="output")
    parser.add_argument("--load", help="Load from files instead of extracting", action="store_true")
    parser.add_argument("--docs", help="Generate documentation", action="store_true")
    parser.add_argument("--query", help="Query the mappings")
    parser.add_argument("--code", help="Generate code for a view")
    parser.add_argument("--language", help="Language for code generation", default="python")
    
    args = parser.parse_args()
    
    print("\nInitializing COPPER Mapper...")
    
    # Initialize mapper
    mapper = CopperMapper(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, space_key=args.space)
    
    # Load from files or extract
    if args.load:
        print(f"Loading from {args.output} directory...")
        if not mapper.load_from_files(args.output):
            print("Error: Failed to load from files.")
            return
    else:
        print("Connecting to Confluence and extracting views and endpoints...")
        if not mapper.initialize():
            print("Error: Failed to initialize. Please check the logs for details.")
            return
        
        # Save to files
        print(f"Saving to {args.output} directory...")
        mapper.save_to_files(args.output)
    
    print(f"Found {len(mapper.views)} views, {len(mapper.endpoints)} endpoints, and {len(mapper.mappings)} mappings.")
    
    # Generate documentation
    if args.docs:
        print("Generating documentation...")
        docs_file = os.path.join(args.output, "documentation.md")
        if mapper.generate_documentation(docs_file):
            print(f"Documentation saved to {docs_file}")
        else:
            print("Error: Failed to generate documentation.")
    
    # Query mappings
    if args.query:
        print(f"Querying: {args.query}")
        response = mapper.query_mappings(args.query)
        print("\nResponse:")
        print("-------")
        print(response)
        print("-------")
    
    # Generate code
    if args.code:
        print(f"Generating {args.language} code for view: {args.code}")
        code = mapper.generate_api_code(args.code, args.language)
        print("\nCode:")
        print("-------")
        print(code)
        print("-------")
    
    print("\nCOPPER Mapper completed.")


if __name__ == "__main__":
    import argparse
    main()
