#!/usr/bin/env python3
"""
Enhanced COPPER Knowledge Assistant
-----------------------------------
Comprehensive solution that loads ALL Confluence content,
intelligently extracts ALL relevant information including tables and image metadata,
and uses advanced relevance scoring to provide accurate answers about COPPER.
"""

import logging
import os
import sys
import json
import re
import time
import concurrent.futures
from datetime import datetime
import threading
import queue
from collections import defaultdict
import hashlib
import pickle

# Confluence imports
import requests
from bs4 import BeautifulSoup, NavigableString
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
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "xyz")  # Target specific space

# Performance and caching settings
MAX_WORKERS = 10  # Increased number of parallel workers
CACHE_DIR = "copper_cache"
RELEVANCE_THRESHOLD = 0.6  # Minimum relevance score to include content (as suggested)


class AdvancedContentExtractor:
    """Advanced extraction of content from Confluence HTML, with special focus on tables and images."""
    
    @staticmethod
    def extract_text_between_tags(soup, tag_name, class_name=None):
        """Extract text from specific tags with optional class filter."""
        results = []
        tags = soup.find_all(tag_name, class_=class_name) if class_name else soup.find_all(tag_name)
        for tag in tags:
            text = tag.get_text().strip()
            if text:
                results.append(text)
        return results
    
    @staticmethod
    def extract_table_with_headers(table_soup):
        """
        Extract a table with proper header-to-data relationships.
        Works with complex table structures including rowspan and colspan.
        """
        # Initialize table data structure
        table_data = {
            "headers": [],
            "rows": [],
            "caption": "",
            "metadata": {}
        }
        
        # Extract caption if available
        caption = table_soup.find('caption')
        if caption:
            table_data["caption"] = caption.get_text().strip()
        
        # Extract headers
        headers = []
        thead = table_soup.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            if header_rows:
                # Process all header rows to handle multi-level headers
                for row in header_rows:
                    header_cells = []
                    for cell in row.find_all(['th', 'td']):
                        header_cells.append({
                            "text": cell.get_text().strip(),
                            "rowspan": int(cell.get('rowspan', 1)),
                            "colspan": int(cell.get('colspan', 1))
                        })
                    if header_cells:
                        headers.append(header_cells)
                
                # For simplicity, flatten multi-level headers
                if headers:
                    flattened_headers = []
                    for row in headers:
                        for cell in row:
                            header_text = cell["text"]
                            if header_text and header_text not in flattened_headers:
                                flattened_headers.append(header_text)
                    table_data["headers"] = flattened_headers
        
        # If no headers found in thead, look for first row with th elements
        if not table_data["headers"]:
            first_row = table_soup.find('tr')
            if first_row:
                header_cells = first_row.find_all('th')
                if header_cells:
                    table_data["headers"] = [cell.get_text().strip() for cell in header_cells]
        
        # If still no headers, check if first row looks like a header
        if not table_data["headers"]:
            first_row = table_soup.find('tr')
            if first_row:
                # Check if it looks like a header row based on formatting or position
                cells = first_row.find_all('td')
                if cells and all(cell.get('style') and ('bold' in cell.get('style') or 'background' in cell.get('style')) for cell in cells):
                    table_data["headers"] = [cell.get_text().strip() for cell in cells]
                elif cells and first_row.parent.name != 'tbody':
                    # First row outside tbody might be a header
                    table_data["headers"] = [cell.get_text().strip() for cell in cells]
        
        # Extract rows
        rows = []
        body_rows = []
        
        # Check for tbody first
        tbody = table_soup.find('tbody')
        if tbody:
            body_rows = tbody.find_all('tr')
        else:
            # If no tbody, get all rows (skipping header row if headers were found)
            all_rows = table_soup.find_all('tr')
            skip_first = len(table_data["headers"]) > 0 and len(all_rows) > 0
            body_rows = all_rows[1:] if skip_first else all_rows
        
        # Process each row
        for row in body_rows:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = []
                for cell in cells:
                    cell_text = cell.get_text().strip()
                    # Check for nested tables and extract them separately
                    nested_tables = cell.find_all('table')
                    if nested_tables:
                        nested_data = []
                        for nested_table in nested_tables:
                            # Remove the nested table from cell text to avoid duplication
                            for tag in cell.find_all('table'):
                                tag.decompose()
                            # Get the clean cell text
                            cell_text = cell.get_text().strip()
                            # Extract the nested table
                            nested_table_data = AdvancedContentExtractor.extract_table_with_headers(nested_table)
                            if nested_table_data["rows"]:
                                nested_data.append(nested_table_data)
                        
                        # Combine cell text with nested table info
                        if nested_data:
                            cell_info = {
                                "text": cell_text,
                                "nested_tables": nested_data,
                                "rowspan": int(cell.get('rowspan', 1)),
                                "colspan": int(cell.get('colspan', 1))
                            }
                            row_data.append(cell_info)
                        else:
                            row_data.append(cell_text)
                    else:
                        row_data.append(cell_text)
                
                if any(cell for cell in row_data if cell):  # Skip empty rows
                    rows.append(row_data)
        
        table_data["rows"] = rows
        
        # Extract any additional metadata about the table
        table_class = table_soup.get('class', [])
        table_id = table_soup.get('id', '')
        if table_class or table_id:
            table_data["metadata"] = {
                "class": table_class,
                "id": table_id
            }
        
        return table_data
    
    @staticmethod
    def format_table_for_context(table_data):
        """Format extracted table data into a text representation."""
        lines = []
        
        # Add caption/title
        if table_data["caption"]:
            lines.append(f"TABLE: {table_data['caption']}")
        elif table_data["metadata"].get("id"):
            lines.append(f"TABLE ID: {table_data['metadata']['id']}")
        else:
            lines.append(f"TABLE:")
        
        # Use headers if available
        if table_data["headers"]:
            headers = table_data["headers"]
            # Calculate column widths
            col_widths = [len(str(h)) for h in headers]
            
            # Update column widths based on data
            for row in table_data["rows"]:
                for i, cell in enumerate(row[:len(col_widths)]):
                    if i < len(col_widths):
                        if isinstance(cell, dict):
                            cell_text = cell["text"]
                        else:
                            cell_text = str(cell)
                        col_widths[i] = max(col_widths[i], min(len(cell_text), 30))  # Limit width to 30 chars
            
            # Format header
            header_row = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
            separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
            lines.append(header_row)
            lines.append(separator)
            
            # Format data rows with headers
            for row in table_data["rows"]:
                # Prepare row data aligned with headers
                row_cells = []
                for i, header in enumerate(headers):
                    if i < len(row):
                        if isinstance(row[i], dict):
                            cell_text = row[i]["text"]
                        else:
                            cell_text = str(row[i])
                        row_cells.append(cell_text.ljust(col_widths[i]))
                    else:
                        row_cells.append("".ljust(col_widths[i]))
                
                lines.append("| " + " | ".join(row_cells) + " |")
        
        else:
            # Table without headers
            # Calculate column widths
            col_count = max(len(row) for row in table_data["rows"]) if table_data["rows"] else 0
            if col_count == 0:
                return "TABLE: [Empty table]"
                
            col_widths = [0] * col_count
            for row in table_data["rows"]:
                for i, cell in enumerate(row[:col_count]):
                    if isinstance(cell, dict):
                        cell_text = cell["text"]
                    else:
                        cell_text = str(cell)
                    col_widths[i] = max(col_widths[i], min(len(cell_text), 30))
            
            # Format rows
            for row in table_data["rows"]:
                row_cells = []
                for i in range(col_count):
                    if i < len(row):
                        if isinstance(row[i], dict):
                            cell_text = row[i]["text"]
                        else:
                            cell_text = str(row[i])
                        row_cells.append(cell_text.ljust(col_widths[i]))
                    else:
                        row_cells.append("".ljust(col_widths[i]))
                
                lines.append("| " + " | ".join(row_cells) + " |")
        
        return "\n".join(lines)
    
    @staticmethod
    def extract_image_context(img_tag, soup):
        """
        Extract comprehensive context for an image, including:
        - Alt text and title
        - Figure captions
        - Surrounding text (preceding and following paragraphs)
        - Parent section title
        - Any image metadata
        """
        context = {
            "alt": img_tag.get('alt', '').strip(),
            "title": img_tag.get('title', '').strip(),
            "src": img_tag.get('src', '').strip(),
            "caption": "",
            "surrounding_text": "",
            "section_title": "",
            "metadata": {}
        }
        
        # Extract figure caption if image is in a figure
        parent_fig = img_tag.find_parent('figure')
        if parent_fig:
            caption = parent_fig.find('figcaption')
            if caption:
                context["caption"] = caption.get_text().strip()
        
        # Look for image in other common patterns
        if not context["caption"]:
            # Check for adjacent div with caption class
            next_div = img_tag.find_next_sibling('div')
            if next_div and next_div.get('class') and any('caption' in cls for cls in next_div.get('class')):
                context["caption"] = next_div.get_text().strip()
            
            # Check for parent div with image+caption pattern
            parent_div = img_tag.find_parent('div')
            if parent_div:
                caption_div = parent_div.find('div', class_=lambda cls: cls and 'caption' in cls)
                if caption_div:
                    context["caption"] = caption_div.get_text().strip()
        
        # Get surrounding text
        prev_text = ""
        next_text = ""
        
        # Previous paragraph or heading
        prev_elem = img_tag.find_previous(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div'])
        if prev_elem and not prev_elem.find('img'):  # Avoid other images
            prev_text = prev_elem.get_text().strip()
        
        # Next paragraph
        next_elem = img_tag.find_next(['p', 'div'])
        if next_elem and not next_elem.find('img'):  # Avoid other images
            next_text = next_elem.get_text().strip()
        
        if prev_text or next_text:
            context["surrounding_text"] = f"{prev_text}\n{next_text}".strip()
        
        # Find section title (nearest preceding heading)
        section_heading = img_tag.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if section_heading:
            context["section_title"] = section_heading.get_text().strip()
        
        # Extract any data attributes that might contain metadata
        for attr, value in img_tag.attrs.items():
            if attr.startswith('data-'):
                context["metadata"][attr] = value
        
        return context
    
    @staticmethod
    def format_image_for_context(image_context):
        """Format extracted image context into a text representation."""
        parts = ["[IMAGE]"]
        
        if image_context["alt"]:
            parts.append(f"Description: {image_context['alt']}")
        elif image_context["title"]:
            parts.append(f"Title: {image_context['title']}")
        
        if image_context["caption"]:
            parts.append(f"Caption: {image_context['caption']}")
        
        if image_context["section_title"]:
            parts.append(f"In section: {image_context['section_title']}")
        
        if image_context["surrounding_text"]:
            parts.append(f"Context: {image_context['surrounding_text']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def extract_content_from_html(html_content, title=""):
        """
        Extract comprehensive content from HTML including tables, images, and structured content.
        
        Args:
            html_content: The HTML content to process
            title: The title of the content
            
        Returns:
            Dict containing processed content elements
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Structure to hold all extracted content
            extracted = {
                "title": title,
                "headings": [],
                "paragraphs": [],
                "tables": [],
                "images": [],
                "code_blocks": [],
                "lists": [],
                "structured_content": [],
                "definitions": [],  # For definition lists and terms
                "raw_text": ""
            }
            
            # Extract all headings with their hierarchy and content
            for heading_level in range(1, 7):
                heading_tag = f'h{heading_level}'
                for heading in soup.find_all(heading_tag):
                    heading_text = heading.get_text().strip()
                    if heading_text:
                        extracted["headings"].append({
                            "level": heading_level,
                            "text": heading_text
                        })
            
            # Extract paragraphs
            for p in soup.find_all('p'):
                # Skip empty paragraphs
                if not p.get_text().strip():
                    continue
                
                # Skip paragraphs that are part of other structures (like figure captions)
                if p.parent.name in ['figure', 'figcaption']:
                    continue
                
                # Get associated heading if possible
                closest_heading = {
                    "text": "",
                    "level": 0
                }
                
                heading = p.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if heading:
                    closest_heading = {
                        "level": int(heading.name[1]),
                        "text": heading.get_text().strip()
                    }
                
                extracted["paragraphs"].append({
                    "text": p.get_text().strip(),
                    "heading": closest_heading
                })
            
            # Extract tables with comprehensive processing
            for table in soup.find_all('table'):
                table_data = AdvancedContentExtractor.extract_table_with_headers(table)
                if table_data["rows"]:  # Only include non-empty tables
                    extracted["tables"].append(table_data)
            
            # Extract images with surrounding context
            for img in soup.find_all('img'):
                image_context = AdvancedContentExtractor.extract_image_context(img, soup)
                if image_context["alt"] or image_context["caption"] or image_context["surrounding_text"]:
                    extracted["images"].append(image_context)
            
            # Extract code blocks
            for pre in soup.find_all('pre'):
                code = pre.find('code')
                if code:
                    # Check for language specification
                    code_class = code.get('class', [])
                    language = ""
                    for cls in code_class:
                        if cls.startswith('language-'):
                            language = cls.replace('language-', '')
                            break
                    
                    # Get the code content
                    code_content = code.get_text().strip()
                    extracted["code_blocks"].append({
                        "language": language,
                        "content": code_content
                    })
                else:
                    # Pre without code tag
                    extracted["code_blocks"].append({
                        "language": "",
                        "content": pre.get_text().strip()
                    })
            
            # Extract lists (ordered and unordered)
            for list_tag in soup.find_all(['ul', 'ol']):
                list_items = []
                for li in list_tag.find_all('li', recursive=False):
                    # Skip empty list items
                    if not li.get_text().strip():
                        continue
                    
                    # Check for nested lists
                    nested_lists = li.find_all(['ul', 'ol'], recursive=False)
                    if nested_lists:
                        # Handle nested list items
                        nested_items = []
                        for nested_list in nested_lists:
                            for nested_li in nested_list.find_all('li'):
                                nested_text = nested_li.get_text().strip()
                                if nested_text:
                                    nested_items.append(nested_text)
                        
                        list_items.append({
                            "text": li.get_text().strip(),
                            "nested_items": nested_items
                        })
                    else:
                        list_items.append({
                            "text": li.get_text().strip()
                        })
                
                if list_items:
                    extracted["lists"].append({
                        "type": list_tag.name,  # "ul" or "ol"
                        "items": list_items
                    })
            
            # Extract definition lists
            for dl in soup.find_all('dl'):
                definitions = []
                
                # Group dt and dd elements
                current_term = None
                
                for child in dl.children:
                    if child.name == 'dt':
                        if current_term and current_term["definition"]:
                            definitions.append(current_term)
                        current_term = {
                            "term": child.get_text().strip(),
                            "definition": ""
                        }
                    elif child.name == 'dd' and current_term:
                        current_term["definition"] = child.get_text().strip()
                
                # Add the last term
                if current_term and current_term["definition"]:
                    definitions.append(current_term)
                
                if definitions:
                    extracted["definitions"].extend(definitions)
            
            # Extract panel content (notes, warnings, info boxes)
            for div in soup.find_all(['div', 'section']):
                if not div.get('class'):
                    continue
                
                class_str = ' '.join(div.get('class', []))
                
                # Look for common panel and message classes
                if any(term in class_str.lower() for term in ['panel', 'note', 'warning', 'info', 'error', 'success', 'aui-message']):
                    # Get panel title if available
                    title_elem = div.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b', '.aui-message-heading'])
                    panel_title = title_elem.get_text().strip() if title_elem else ""
                    
                    # Determine panel type from class
                    panel_type = ""
                    for type_name in ['note', 'info', 'warning', 'error', 'success', 'tip']:
                        if type_name in class_str.lower():
                            panel_type = type_name
                            break
                    
                    if not panel_type:
                        panel_type = "panel"
                    
                    # Get panel content
                    # Remove the title element to avoid duplication
                    if title_elem:
                        title_elem.extract()
                    
                    panel_content = div.get_text().strip()
                    
                    if panel_content:
                        extracted["structured_content"].append({
                            "type": panel_type.upper(),
                            "title": panel_title,
                            "content": panel_content
                        })
            
            # Extract raw text for general search and context
            extracted["raw_text"] = soup.get_text().strip()
            
            return extracted
        
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            # Return minimal structure with error message
            return {
                "title": title,
                "headings": [],
                "paragraphs": [],
                "tables": [],
                "images": [],
                "code_blocks": [],
                "lists": [],
                "structured_content": [],
                "definitions": [],
                "raw_text": f"Error extracting content: {str(e)}"
            }
    
    @staticmethod
    def format_for_context(extracted_content, include_all=False):
        """
        Format extracted content into a well-structured text representation.
        
        Args:
            extracted_content: The dictionary of extracted content elements
            include_all: Whether to include all content or a summary
            
        Returns:
            Formatted string with all content elements
        """
        sections = []
        
        # Add title
        if extracted_content["title"]:
            sections.append(f"# {extracted_content['title']}")
        
        # Create a structured document with headings
        current_section = ""
        current_heading = {"level": 0, "text": ""}
        
        # Sort headings by their position in the document to maintain structure
        all_headings = extracted_content["headings"].copy()
        
        # Add paragraphs with their headings
        for paragraph in extracted_content["paragraphs"]:
            heading = paragraph["heading"]
            
            # If we've moved to a new heading section, add the previous section
            if heading != current_heading and current_section:
                sections.append(current_section)
                current_section = ""
            
            # Update current heading if needed
            if heading != current_heading:
                current_heading = heading
                if heading["text"]:
                    current_section = f"{'#' * (heading['level'] + 1)} {heading['text']}\n\n"
            
            # Add paragraph text to the current section
            current_section += paragraph["text"] + "\n\n"
        
        # Add the last section if it exists
        if current_section:
            sections.append(current_section)
        
        # Add tables with proper formatting
        if extracted_content["tables"]:
            for table_data in extracted_content["tables"]:
                formatted_table = AdvancedContentExtractor.format_table_for_context(table_data)
                sections.append(f"\n{formatted_table}\n")
        
        # Add lists
        for list_data in extracted_content["lists"]:
            list_type = list_data["type"]
            list_text = []
            
            for i, item in enumerate(list_data["items"], 1):
                if list_type == "ol":
                    prefix = f"{i}. "
                else:
                    prefix = "• "
                
                if "nested_items" in item:
                    list_text.append(f"{prefix}{item['text']}")
                    for nested in item["nested_items"]:
                        list_text.append(f"  - {nested}")
                else:
                    list_text.append(f"{prefix}{item['text']}")
            
            if list_text:
                sections.append("\n".join(list_text))
        
        # Add definitions
        if extracted_content["definitions"]:
            definitions_text = []
            for definition in extracted_content["definitions"]:
                definitions_text.append(f"{definition['term']}: {definition['definition']}")
            
            if definitions_text:
                sections.append("Definitions:\n" + "\n".join(definitions_text))
        
        # Add code blocks
        for code_block in extracted_content["code_blocks"]:
            language = code_block["language"]
            content = code_block["content"]
            
            if language:
                sections.append(f"```{language}\n{content}\n```")
            else:
                sections.append(f"```\n{content}\n```")
        
        # Add image information
        for image in extracted_content["images"]:
            image_text = AdvancedContentExtractor.format_image_for_context(image)
            sections.append(image_text)
        
        # Add structured content (panels, notes, etc.)
        for content in extracted_content["structured_content"]:
            formatted = f"[{content['type']}]"
            if content["title"]:
                formatted += f" {content['title']}"
            formatted += f"\n{content['content']}"
            sections.append(formatted)
        
        return "\n\n".join(sections)


class PersistentCache:
    """Thread-safe persistent cache for storing API responses and extracted content."""
    
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.Lock()
        
        # Create subdirectories for different cache types
        self.api_cache_dir = os.path.join(cache_dir, "api_responses")
        self.content_cache_dir = os.path.join(cache_dir, "page_content")
        self.metadata_cache_dir = os.path.join(cache_dir, "metadata")
        
        os.makedirs(self.api_cache_dir, exist_ok=True)
        os.makedirs(self.content_cache_dir, exist_ok=True)
        os.makedirs(self.metadata_cache_dir, exist_ok=True)
        
        # In-memory cache for frequently used items
        self.memory_cache = {}
        self.memory_cache_size = 100
    
    def _get_cache_key(self, key_components):
        """Generate a stable cache key from components."""
        if isinstance(key_components, str):
            key_str = key_components
        else:
            key_str = json.dumps(key_components, sort_keys=True)
        
        # Generate a hash of the key for filenames
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key, cache_type="api"):
        """Get the file path for a cache item."""
        cache_dir = self.api_cache_dir
        if cache_type == "content":
            cache_dir = self.content_cache_dir
        elif cache_type == "metadata":
            cache_dir = self.metadata_cache_dir
        
        return os.path.join(cache_dir, f"{key}.pickle")
    
    def get(self, key, cache_type="api"):
        """
        Get a value from the cache.
        
        Args:
            key: The cache key (string or serializable object)
            cache_type: Type of cache ("api", "content", or "metadata")
            
        Returns:
            The cached value or None if not found
        """
        cache_key = self._get_cache_key(key)
        
        # Check memory cache first
        memory_key = f"{cache_type}:{cache_key}"
        with self.lock:
            if memory_key in self.memory_cache:
                return self.memory_cache[memory_key]
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key, cache_type)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                
                # Add to memory cache
                with self.lock:
                    if len(self.memory_cache) >= self.memory_cache_size:
                        # Remove a random item if full
                        self.memory_cache.pop(next(iter(self.memory_cache)))
                    self.memory_cache[memory_key] = data
                
                return data
            except Exception as e:
                logger.error(f"Error reading from cache: {str(e)}")
        
        return None
    
    def set(self, key, value, cache_type="api"):
        """
        Store a value in the cache.
        
        Args:
            key: The cache key (string or serializable object)
            value: The value to cache
            cache_type: Type of cache ("api", "content", or "metadata")
        """
        cache_key = self._get_cache_key(key)
        
        # Add to memory cache
        memory_key = f"{cache_type}:{cache_key}"
        with self.lock:
            if len(self.memory_cache) >= self.memory_cache_size:
                # Remove a random item if full
                self.memory_cache.pop(next(iter(self.memory_cache)))
            self.memory_cache[memory_key] = value
        
        # Save to file cache
        cache_path = self._get_cache_path(cache_key, cache_type)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
    
    def clear(self, cache_type=None):
        """
        Clear the cache.
        
        Args:
            cache_type: Type of cache to clear or None for all
        """
        with self.lock:
            # Clear memory cache
            if cache_type:
                keys_to_remove = [k for k in self.memory_cache if k.startswith(f"{cache_type}:")]
                for k in keys_to_remove:
                    self.memory_cache.pop(k, None)
            else:
                self.memory_cache.clear()
        
        # Clear file cache
        if cache_type == "api" or cache_type is None:
            for f in os.listdir(self.api_cache_dir):
                os.remove(os.path.join(self.api_cache_dir, f))
        
        if cache_type == "content" or cache_type is None:
            for f in os.listdir(self.content_cache_dir):
                os.remove(os.path.join(self.content_cache_dir, f))
        
        if cache_type == "metadata" or cache_type is None:
            for f in os.listdir(self.metadata_cache_dir):
                os.remove(os.path.join(self.metadata_cache_dir, f))


class ConfluenceClient:
    """Enhanced client for Confluence REST API with comprehensive error handling and improved caching."""
    
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
        self.api_url = f"{self.base_url}/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "COPPER-AI-Python-Agent"
        }
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update(self.headers)
        self.session.verify = False  # As requested: verify=False for SSL
        
        # Set up requests with retries
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', retry_adapter)
        self.session.mount('https://', retry_adapter)
        
        # Initialize cache
        self.cache = PersistentCache()
        
        # Default timeout (increased for large page downloads)
        self.timeout = 60
        
        logger.info(f"Initialized enhanced Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            
            # Try to get space info first (faster than content)
            response = self.session.get(
                f"{self.api_url}/space",
                params={"limit": 1},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info("Connection to Confluence successful!")
                return True
            else:
                logger.warning(f"Unexpected status code during connection test: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def _make_api_request(self, url, method="GET", params=None, data=None, use_cache=True):
        """
        Make an API request with automatic caching.
        
        Args:
            url: The API URL
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: POST data
            use_cache: Whether to use cache for GET requests
            
        Returns:
            The JSON response or None on error
        """
        # Only cache GET requests
        use_cache = use_cache and method.upper() == "GET"
        
        if use_cache:
            # Generate a cache key from the request details
            cache_key = {
                "url": url,
                "params": params or {}
            }
            
            # Check cache first
            cached_response = self.cache.get(cache_key, "api")
            if cached_response:
                return cached_response
        
        # Make the request
        try:
            method_func = getattr(self.session, method.lower())
            
            response = method_func(
                url,
                params=params,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse and cache response for GET requests
            if response.text.strip():
                response_data = response.json()
                
                if use_cache:
                    self.cache.set(cache_key, response_data, "api")
                
                return response_data
            
            return None
            
        except requests.RequestException as e:
            logger.error(f"API request failed for {url}: {str(e)}")
            # Log response content if available
            if hasattr(e, 'response') and e.response:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    logger.error(f"Response: {e.response.text[:1000]}")
                except:
                    pass
            return None
    
    def get_space_info(self, space_key):
        """
        Get information about a specific Confluence space.
        
        Args:
            space_key: The space key to get information for
            
        Returns:
            Space information or None if not found
        """
        url = f"{self.api_url}/space/{space_key}"
        
        # Include homepage and metadata in expansion
        params = {
            "expand": "homepage,description.view,metadata"
        }
        
        return self._make_api_request(url, params=params)
    
    def get_all_pages_in_space(self, space_key, include_archived=False, batch_size=100):
        """
        Get ALL pages in a Confluence space using efficient pagination.
        
        Args:
            space_key: The space key to get all pages from
            include_archived: Whether to include archived pages
            batch_size: Number of results per request (max 100)
            
        Returns:
            List of page objects with basic information
        """
        all_pages = []
        start = 0
        has_more = True
        
        logger.info(f"Fetching all pages from space: {space_key}")
        
        # Check if we have a cached result first
        cache_key = f"all_pages_{space_key}_{include_archived}"
        cached_pages = self.cache.get(cache_key, "metadata")
        if cached_pages:
            logger.info(f"Using {len(cached_pages)} cached pages for space {space_key}")
            return cached_pages
        
        # If not in cache, fetch all pages
        while has_more:
            logger.info(f"Fetching pages batch from start={start}")
            
            params = {
                "spaceKey": space_key,
                "expand": "history.lastUpdated,metadata.labels",
                "status": "current" if not include_archived else "any",
                "limit": batch_size,
                "start": start
            }
            
            response_data = self._make_api_request(
                f"{self.api_url}/content", 
                params=params
            )
            
            if not response_data:
                logger.warning(f"Failed to get response for pages at start={start}")
                break
            
            results = response_data.get("results", [])
            all_pages.extend(results)
            
            # Check if there are more pages
            if "size" in response_data and "limit" in response_data:
                if response_data["size"] < response_data["limit"]:
                    has_more = False
                else:
                    start += batch_size
            else:
                has_more = False
            
            logger.info(f"Fetched {len(results)} pages, total so far: {len(all_pages)}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Cache the results
        self.cache.set(cache_key, all_pages, "metadata")
        
        logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
        return all_pages
    
    def get_page_with_content(self, page_id):
        """
        Get a page with its content and metadata.
        
        Args:
            page_id: The ID of the page
            
        Returns:
            Page object with content or None if not found
        """
        # Check cache first
        cache_key = f"page_with_content_{page_id}"
        cached_page = self.cache.get(cache_key, "content")
        if cached_page:
            return cached_page
        
        # Get page with content and expanded properties
        url = f"{self.api_url}/content/{page_id}"
        params = {
            "expand": "body.storage,metadata.labels,history.lastUpdated,version,children.attachment"
        }
        
        page = self._make_api_request(url, params=params)
        if not page:
            return None
        
        # Cache and return
        self.cache.set(cache_key, page, "content")
        return page
    
    def search_content(self, cql, limit=100, expand="body.storage", all_results=False):
        """
        Search for content using Confluence Query Language (CQL).
        
        Args:
            cql: Confluence Query Language query string
            limit: Maximum number of results per request
            expand: Fields to expand in results
            all_results: Whether to fetch all matching results through pagination
            
        Returns:
            List of search results or empty list on error
        """
        url = f"{self.api_url}/content/search"
        
        if all_results:
            # Get all results via pagination
            all_results = []
            start = 0
            has_more = True
            
            while has_more:
                params = {
                    "cql": cql,
                    "limit": limit,
                    "start": start,
                    "expand": expand
                }
                
                response_data = self._make_api_request(url, params=params)
                
                if not response_data:
                    break
                
                results = response_data.get("results", [])
                all_results.extend(results)
                
                # Check if there are more pages
                if "size" in response_data and "limit" in response_data:
                    if response_data["size"] < response_data["limit"]:
                        has_more = False
                    else:
                        start += response_data["size"]
                else:
                    has_more = False
                
                # Avoid rate limiting
                time.sleep(0.1)
            
            return all_results
        else:
            # Get single page of results
            params = {
                "cql": cql,
                "limit": limit,
                "expand": expand
            }
            
            response_data = self._make_api_request(url, params=params)
            
            if not response_data:
                return []
            
            return response_data.get("results", [])
    
    def get_content_by_id(self, content_id, expand="body.storage"):
        """
        Get content by ID with expanded fields.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma separated list of fields to expand
            
        Returns:
            Content object or None if not found
        """
        url = f"{self.api_url}/content/{content_id}"
        params = {"expand": expand}
        
        return self._make_api_request(url, params=params)
    
    def extract_and_process_page_content(self, page_id):
        """
        Get a page by ID, extract its content, and process it for searching and analysis.
        
        Args:
            page_id: The ID of the page
            
        Returns:
            Dict with processed content or None if failed
        """
        # Check cache first
        cache_key = f"processed_content_{page_id}"
        cached_content = self.cache.get(cache_key, "content")
        if cached_content:
            return cached_content
        
        # Get page with content
        page = self.get_page_with_content(page_id)
        if not page:
            return None
        
        # Extract HTML content
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        if not html_content:
            return None
        
        # Get metadata
        title = page.get("title", "")
        updated_date = page.get("history", {}).get("lastUpdated", {}).get("when", "")
        labels = [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
        
        # Process content with advanced extractor
        processed_content = AdvancedContentExtractor.extract_content_from_html(html_content, title)
        
        # Add metadata
        processed_content["page_id"] = page_id
        processed_content["url"] = f"{self.base_url}/pages/viewpage.action?pageId={page_id}"
        processed_content["labels"] = labels
        processed_content["updated_date"] = updated_date
        
        # Cache the processed content
        self.cache.set(cache_key, processed_content, "content")
        
        return processed_content


class GeminiAssistant:
    """Enhanced class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        """Initialize the Gemini assistant."""
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")
    
    def generate_response(self, prompt, context=None, temperature=0.2):
        """
        Generate a response from Gemini based on the prompt and context.
        
        Args:
            prompt: The user's question or prompt
            context: Context information (from Confluence)
            temperature: Temperature parameter for generation (0.0-1.0)
            
        Returns:
            The generated response
        """
        logger.info(f"Generating response for prompt: {prompt}")
        
        try:
            # Enhanced system prompt tailored for the task
            system_prompt = """
            You are the COPPER Knowledge Assistant, an expert on database views, REST APIs, and system integration.
            
            Your personality:
            - Conversational, friendly, and helpful - like a knowledgeable colleague
            - Clear and precise in technical explanations
            - You simplify complex concepts without oversimplifying
            - You recognize patterns and connections across documentation
            
            Your knowledge:
            - Deep expertise in the COPPER database and API system
            - Understanding of database views, table structures, and API endpoints
            - Familiarity with database-to-API mapping patterns
            
            When answering questions:
            1. Be direct and thorough - answer the specific question first, then provide helpful context
            2. Include specific details from the documentation rather than general statements
            3. If tables are relevant to the answer, include their structure and explain relationships
            4. Reference specific API endpoints, parameters, or database fields when applicable
            5. Format your responses for readability, using lists and proper spacing
            6. For acronyms or special terms, provide definitions or explanations
            7. Look for the "full form" of acronyms like COPPER if asked
            8. Thoroughly explain API endpoints when requested
            
            Remember that you have been provided with extensive documentation about COPPER.
            Use this information to give precise, helpful answers. If you're not sure about something,
            acknowledge what you don't know rather than making up information.
            """
            
            # Craft the full prompt with context
            full_prompt = system_prompt + "\n\n"
            
            if context:
                # Handle large contexts intelligently by summarizing when needed
                if len(context) > 28000:  # Generous limit for context window
                    logger.warning(f"Context too large ({len(context)} chars), performing intelligent trimming...")
                    
                    # Split context into manageable chunks by page/section
                    context_sections = context.split("\n\n--- FROM:")
                    
                    # Keep introduction and essential parts
                    trimmed_context = context_sections[0]
                    
                    # Add most relevant sections
                    for section in context_sections[1:]:
                        if len(trimmed_context) + len(section) + 10 < 28000:
                            trimmed_context += "\n\n--- FROM:" + section
                        else:
                            # We've reached our limit
                            break
                    
                    context = trimmed_context
                    logger.info(f"Trimmed context to {len(context)} chars")
                
                full_prompt += "CONTEXT FROM COPPER DOCUMENTATION:\n" + context + "\n\n"
                
            full_prompt += f"USER QUESTION: {prompt}\n\nResponse:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
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
                return "I couldn't find a clear answer to that question in the COPPER documentation. Would you like me to look for related information instead?"
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered a technical issue while processing your question. Please try asking in a different way or ask another question."
    
    def categorize_question(self, question):
        """
        Analyze the question to determine its category for better context selection.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with question category and key terms
        """
        # Define common question categories and associated terms
        categories = {
            "definition": ["what is", "what are", "define", "meaning", "full form", "acronym", "stand for", "definition"],
            "api_endpoints": ["api endpoint", "endpoints", "rest api", "api", "request", "response", "url", "http"],
            "database": ["table", "view", "column", "field", "schema", "database", "query", "sql"],
            "mapping": ["map", "mapping", "relationship", "connect", "link", "correspond", "translation"],
            "configuration": ["config", "configure", "setup", "setting", "parameter", "property"],
            "tutorial": ["how to", "tutorial", "guide", "steps", "procedure", "example"]
        }
        
        # Normalize question
        question_lower = question.lower()
        
        # Score each category
        category_scores = {}
        for category, terms in categories.items():
            score = 0
            for term in terms:
                if term in question_lower:
                    score += 1
            category_scores[category] = score
        
        # Get top categories (may be more than one)
        max_score = max(category_scores.values())
        if max_score > 0:
            top_categories = [c for c, s in category_scores.items() if s == max_score]
        else:
            # Default to general question if no clear category
            top_categories = ["general"]
        
        # Extract potential key terms (nouns, technical terms, etc.)
        key_terms = []
        
        # Look for quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', question) + re.findall(r"'([^']+)'", question)
        if quoted_terms:
            key_terms.extend(quoted_terms)
        
        # Look for technical terms using common patterns
        # Acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', question)
        if acronyms:
            key_terms.extend(acronyms)
        
        # CamelCase terms
        camel_case = re.findall(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b', question)
        if camel_case:
            key_terms.extend(camel_case)
        
        # Terms with underscores
        underscored = re.findall(r'\b\w+_\w+\b', question)
        if underscored:
            key_terms.extend(underscored)
        
        return {
            "categories": top_categories,
            "key_terms": key_terms,
            "original_question": question
        }


class CopperAssistant:
    """Main class that coordinates between Confluence and Gemini."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None):
        """
        Initialize the COPPER Assistant.
        
        Args:
            confluence_url: The Confluence URL
            confluence_username: Confluence username
            confluence_api_token: Confluence API token
            space_key: The Confluence space key to focus on
        """
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.gemini = GeminiAssistant()
        self.space_key = space_key
        self.space_pages = []
        self.processed_pages = {}
        logger.info(f"Initialized COPPER Assistant targeting space: {space_key or 'all spaces'}")
    
    def initialize(self):
        """Initialize the assistant by testing connections and loading content."""
        logger.info("Initializing COPPER Assistant...")
        
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and URL.")
            return False
        
        logger.info("Connection to Confluence successful.")
        
        # Load space information
        if self.space_key:
            space_info = self.confluence.get_space_info(self.space_key)
            if not space_info:
                logger.error(f"Failed to get information for space {self.space_key}.")
                return False
            
            logger.info(f"Successfully loaded information for space {self.space_key}: {space_info.get('name', 'Unknown')}")
            
            # Load all pages in the space
            self.load_all_space_content()
        
        return True
    
    def load_all_space_content(self):
        """Load ALL pages in the specified space."""
        if not self.space_key:
            logger.error("No space key specified. Please provide a space key.")
            return
        
        logger.info(f"Loading all pages in space {self.space_key}...")
        
        # Get all pages in the space
        self.space_pages = self.confluence.get_all_pages_in_space(self.space_key)
        
        logger.info(f"Loaded metadata for {len(self.space_pages)} pages from space {self.space_key}")
    
    def preprocess_all_pages(self):
        """
        Extract and process content from all pages.
        This can be done in advance to speed up queries.
        """
        if not self.space_pages:
            logger.error("No pages loaded. Call load_all_space_content() first.")
            return
        
        logger.info(f"Preprocessing content from {len(self.space_pages)} pages...")
        
        # Use a thread pool to process pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all page processing tasks
            future_to_page = {
                executor.submit(self.confluence.extract_and_process_page_content, page['id']): page['id']
                for page in self.space_pages
            }
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_page)):
                page_id = future_to_page[future]
                try:
                    processed_content = future.result()
                    if processed_content:
                        self.processed_pages[page_id] = processed_content
                except Exception as e:
                    logger.error(f"Error processing page {page_id}: {str(e)}")
                
                # Log progress periodically
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.space_pages)} pages")
        
        logger.info(f"Successfully preprocessed {len(self.processed_pages)} pages")
    
    def _fetch_page_content_batch(self, page_ids):
        """
        Fetch content for a batch of pages in parallel.
        
        Args:
            page_ids: List of page IDs to fetch
            
        Returns:
            Dictionary mapping page IDs to their processed content
        """
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all content extraction tasks
            future_to_page = {
                executor.submit(self.confluence.extract_and_process_page_content, page_id): page_id
                for page_id in page_ids
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_id = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        results[page_id] = content
                except Exception as e:
                    logger.error(f"Error fetching content for page {page_id}: {str(e)}")
        
        return results
    
    def find_relevance_to_query(self, query, page_content):
        """
        Calculate relevance score between a query and page content.
        
        Args:
            query: The user's query
            page_content: Dict with processed page content
            
        Returns:
            Float relevance score between 0.0 and 1.0
        """
        # Categorize the question for better matching
        question_info = self.gemini.categorize_question(query)
        categories = question_info["categories"]
        key_terms = question_info["key_terms"]
        
        # Extract useful content fields
        title = page_content.get("title", "").lower()
        headings = [h["text"].lower() for h in page_content.get("headings", [])]
        raw_text = page_content.get("raw_text", "").lower()
        labels = [label.lower() for label in page_content.get("labels", [])]
        
        # Prepare query terms
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Start with a base score
        score = 0.0
        
        # Check for exact query matches (highest priority)
        if query_lower in raw_text:
            score += 0.4
        
        # Check for key terms (high priority)
        for term in key_terms:
            term_lower = term.lower()
            # Term in title is very important
            if term_lower in title:
                score += 0.3
            # Term in headings is important
            if any(term_lower in heading for heading in headings):
                score += 0.2
            # Term in text
            if term_lower in raw_text:
                score += 0.1
        
        # Check for query words (medium priority)
        words_in_title = sum(1 for word in query_words if word in title)
        title_match_ratio = words_in_title / len(query_words) if query_words else 0
        score += title_match_ratio * 0.2
        
        # For specific question categories, check for relevant content
        if "definition" in categories:
            # Look for definition patterns
            definition_patterns = [
                r'\b(stands for|is an acronym for|is short for)\b',
                r'\b(is|are|means|refers to|defined as)\b.*\b(a|an|the)\b',
                r':.*\.',  # Colon followed by definition
                r'=.*\.'   # Equals sign followed by definition
            ]
            
            if any(re.search(pattern, raw_text) for pattern in definition_patterns):
                score += 0.2
        
        elif "api_endpoints" in categories:
            # Check for API endpoint references
            api_patterns = [
                r'/api/',
                r'(endpoint|rest|http|api)',
                r'(get|post|put|delete)\s+request',
                r'(curl|fetch|request)'
            ]
            
            if any(re.search(pattern, raw_text, re.IGNORECASE) for pattern in api_patterns):
                score += 0.2
        
        # Check for relevant tables if query suggests database or structure interest
        if "database" in categories or "mapping" in categories:
            tables = page_content.get("tables", [])
            if tables:
                table_count = len(tables)
                score += min(0.2, table_count * 0.05)  # Up to 0.2 for tables
        
        # Check page labels for relevance
        if "copper" in labels:
            score += 0.1
        
        # Normalize the score to be between 0 and 1
        return min(1.0, score)
    
    def search_for_content(self, query):
        """
        Search for relevant content across all pages.
        
        Args:
            query: The user's query
            
        Returns:
            List of (page, relevance_score) tuples sorted by relevance
        """
        if not self.space_pages:
            logger.error("No pages loaded. Call load_all_space_content() first.")
            return []
        
        logger.info(f"Searching for content related to: {query}")
        
        # Step 1: Perform initial filtering based on page metadata
        # This helps focus detailed analysis on promising candidates
        candidate_pages = []
        
        # Extract key terms from query
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Try direct Confluence search via CQL for faster initial filtering
        # This leverages Confluence's built-in search index
        try:
            search_terms = " OR ".join(f'"{word}"' for word in query_words if len(word) > 3)
            cql = f'space="{self.space_key}" AND text ~ ({search_terms})'
            
            search_results = self.confluence.search_content(
                cql=cql,
                limit=30,
                expand="",  # No need for content in initial filter
                all_results=False
            )
            
            # Create a set of page IDs from search results for quick lookup
            search_page_ids = {result.get("id") for result in search_results}
            
            # Add search results to candidates
            for page in self.space_pages:
                page_id = page.get("id")
                title = page.get("title", "").lower()
                
                # Give high priority to pages found by Confluence search
                if page_id in search_page_ids:
                    candidate_pages.append(page)
                # Also include pages with query terms in title
                elif any(word in title for word in query_words):
                    candidate_pages.append(page)
            
            logger.info(f"Initial filtering found {len(candidate_pages)} candidate pages")
            
            # If few candidates found, include more pages
            if len(candidate_pages) < 15:
                # Add pages with title suggestions
                for page in self.space_pages:
                    if page not in candidate_pages:
                        title = page.get("title", "").lower()
                        # Check for partial title matches
                        if any(word[:4] in title for word in query_words if len(word) > 4):
                            candidate_pages.append(page)
                
                logger.info(f"Expanded to {len(candidate_pages)} candidate pages")
            
            # Backup: if still few candidates, take most recently updated pages
            if len(candidate_pages) < 10:
                recent_pages = sorted(
                    self.space_pages, 
                    key=lambda p: p.get("history", {}).get("lastUpdated", {}).get("when", "2000-01-01"),
                    reverse=True
                )[:20]
                
                for page in recent_pages:
                    if page not in candidate_pages:
                        candidate_pages.append(page)
                
                logger.info(f"Added recent pages, now have {len(candidate_pages)} candidates")
            
        except Exception as e:
            logger.error(f"Error in initial content filtering: {str(e)}")
            # Fall back to using all pages if search fails
            candidate_pages = self.space_pages[:50]  # Limit to first 50 for performance
        
        # Step 2: Fetch content for candidates in parallel
        logger.info(f"Fetching content for {len(candidate_pages)} candidate pages...")
        
        # Fetch content for candidates
        page_ids = [page.get("id") for page in candidate_pages]
        candidate_contents = self._fetch_page_content_batch(page_ids)
        
        logger.info(f"Successfully fetched content for {len(candidate_contents)} pages")
        
        # Step 3: Score all candidates for relevance
        scored_pages = []
        
        for page_id, content in candidate_contents.items():
            relevance = self.find_relevance_to_query(query, content)
            
            # Include pages with sufficient relevance
            if relevance >= RELEVANCE_THRESHOLD:
                scored_pages.append((content, relevance))
        
        # Sort by relevance score (descending)
        scored_pages.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(scored_pages)} pages above relevance threshold {RELEVANCE_THRESHOLD}")
        
        return scored_pages
    
    def format_content_for_context(self, relevant_pages):
        """
        Format relevant pages into context for the assistant.
        
        Args:
            relevant_pages: List of (page_content, relevance) tuples
            
        Returns:
            Formatted context string
        """
        if not relevant_pages:
            return "No relevant content found in the documentation."
        
        context_parts = []
        
        # Add a header
        context_parts.append(f"I found {len(relevant_pages)} relevant pages in the documentation:")
        
        # Process each page in order of relevance
        for i, (page, relevance) in enumerate(relevant_pages, 1):
            # Format page content appropriately
            page_title = page.get("title", "Untitled")
            page_url = page.get("url", "")
            
            # Start with page header
            page_context = [f"--- FROM: {page_title} ---"]
            
            # Include page content structured for readability
            formatted_content = AdvancedContentExtractor.format_for_context(page)
            page_context.append(formatted_content)
            
            # Add source reference
            page_context.append(f"Source: {page_url}")
            
            # Add to overall context
            context_parts.append("\n\n".join(page_context))
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, question):
        """
        Answer a question using all available content.
        
        Args:
            question: The user's question
            
        Returns:
            The generated response
        """
        logger.info(f"Processing question: {question}")
        
        # Search for relevant content
        start_time = time.time()
        relevant_pages = self.search_for_content(question)
        
        # Check if we found relevant content
        if not relevant_pages:
            logger.warning("No relevant content found for the question.")
            return "I couldn't find specific information about that in the COPPER documentation. Could you try rephrasing your question or ask about a related topic?"
        
        # Format content for context
        context = self.format_content_for_context(relevant_pages)
        
        # Generate response using Gemini
        response = self.gemini.generate_response(question, context)
        
        end_time = time.time()
        logger.info(f"Generated response in {end_time - start_time:.2f} seconds")
        
        return response


def main():
    """Main entry point for the COPPER Assistant."""
    logger.info("Starting COPPER Assistant")
    
    # Check for required environment variables
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN:
        logger.error("Missing Confluence credentials. Please set CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN environment variables.")
        print("Error: Missing Confluence credentials. Please set the required environment variables.")
        return
    
    # Show startup message
    print("\n========================================")
    print("      COPPER Knowledge Assistant")
    print("========================================")
    print("\nInitializing...")
    print("Connecting to Confluence...")
    
    # Initialize the assistant
    assistant = CopperAssistant(
        confluence_url=CONFLUENCE_URL,
        confluence_username=CONFLUENCE_USERNAME,
        confluence_api_token=CONFLUENCE_API_TOKEN,
        space_key=CONFLUENCE_SPACE
    )
    
    if not assistant.initialize():
        logger.error("Failed to initialize COPPER Assistant.")
        print("Error: Failed to initialize. Please check the logs for details.")
        return
    
    print(f"Connected to Confluence space: {CONFLUENCE_SPACE}")
    print(f"Loading information from {len(assistant.space_pages)} pages...")
    
    # Start interactive loop
    print("\nCOPPER Knowledge Assistant is ready!")
    print("Ask anything about COPPER database views, APIs, or mappings.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nThank you for using the COPPER Knowledge Assistant. Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nSearching documentation...")
            
            start_time = time.time()
            answer = assistant.answer_question(user_input)
            end_time = time.time()
            
            print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
            print("-----------------------------------------------")
            print(answer)
            print("-----------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nSession terminated. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"\nSorry, I encountered an error: {str(e)}")
            print("Let's try another question.")
    
    print("\nCOPPER Assistant session ended.")


if __name__ == "__main__":
    main()
