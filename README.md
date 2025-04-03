#!/usr/bin/env python3
"""
Ultimate COPPER Knowledge Assistant
-----------------------------------
A comprehensive, robust solution that extracts ALL information from Confluence pages
including tables, images, links, and follows references to ensure complete knowledge.
"""

import os
import sys
import logging
import json
import time
import requests
import re
import pickle
import hashlib
import base64
import queue
from urllib.parse import urljoin, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup, NavigableString, Tag
import threading
from collections import defaultdict
import tempfile

# For Gemini
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "xyz")

# Cache directories
CACHE_DIR = "copper_cache"
PAGE_INDEX_FILE = os.path.join(CACHE_DIR, f"{CONFLUENCE_SPACE}_page_index.pickle")
RAW_CONTENT_DIR = os.path.join(CACHE_DIR, "raw_content")
PROCESSED_CONTENT_DIR = os.path.join(CACHE_DIR, "processed_content")
REFERENCED_CONTENT_DIR = os.path.join(CACHE_DIR, "referenced_content")
IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, "images")

# Create cache directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_CONTENT_DIR, exist_ok=True)
os.makedirs(PROCESSED_CONTENT_DIR, exist_ok=True)
os.makedirs(REFERENCED_CONTENT_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# Set global constants
MAX_RESULTS = 30
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
MAX_LINK_DEPTH = 2  # How many links deep to follow
MAX_LINKS_PER_PAGE = 5  # Maximum number of links to follow per page
TABLE_EXTRACTION_METHODS = 3  # Number of different methods to try for table extraction


class ComprehensiveContentExtractor:
    """Extract ALL content from Confluence pages including tables, images, and links."""
    
    @staticmethod
    def extract_title(soup):
        """Extract the page title."""
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        
        # Try to find the main heading
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        return "Untitled Page"
    
    @staticmethod
    def extract_headings(soup):
        """Extract all headings with their hierarchy."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    "level": level,
                    "text": heading.get_text().strip(),
                    "id": heading.get('id', '')
                })
        return headings
    
    @staticmethod
    def extract_tables(soup):
        """
        Extract tables using multiple methods to ensure all table data is captured.
        Returns list of extracted tables with their contents.
        """
        all_tables = []
        
        # Method 1: Standard HTML table extraction
        for table_idx, table in enumerate(soup.find_all('table')):
            # Get table caption or create a default one
            caption = ""
            caption_tag = table.find('caption')
            if caption_tag:
                caption = caption_tag.get_text().strip()
            
            if not caption:
                # Try to find a preceding heading as caption
                prev_heading = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev_heading and prev_heading.get_text().strip():
                    caption = f"Table related to: {prev_heading.get_text().strip()}"
                else:
                    caption = f"Table {table_idx + 1}"
            
            # Extract headers
            headers = []
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # If no thead, check if first row looks like a header
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    # Check if row has th elements or td elements with header styling
                    if first_row.find('th') or any('header' in td.get('class', '') for td in first_row.find_all('td')):
                        headers = [cell.get_text().strip() for cell in first_row.find_all(['th', 'td'])]
            
            # Extract rows
            rows = []
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    row = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
            else:
                # If no tbody, get all rows (excluding header if found)
                all_rows = table.find_all('tr')
                start_idx = 1 if headers and len(all_rows) > 0 else 0
                for tr in all_rows[start_idx:]:
                    row = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
            
            # Add table to results
            all_tables.append({
                "caption": caption,
                "headers": headers,
                "rows": rows,
                "extraction_method": "html_table"
            })
        
        # Method 2: Confluence table macros extraction
        for table_macro in soup.find_all('div', class_=lambda c: c and 'table-macro' in c):
            caption = ""
            title_div = table_macro.find('div', class_='confluenceTh')
            if title_div:
                caption = title_div.get_text().strip()
            
            rows = []
            for tr in table_macro.find_all(['tr', 'div'], class_=lambda c: c and 'confluenceTr' in c):
                row = []
                for cell in tr.find_all(['td', 'div'], class_=lambda c: c and ('confluenceTd' in c or 'confluenceTh' in c)):
                    row.append(cell.get_text().strip())
                if row:
                    rows.append(row)
            
            # Try to extract headers (first row if it looks different)
            headers = []
            if rows and 'confluenceTh' in str(table_macro):
                headers = rows[0]
                rows = rows[1:]
            
            # Add table to results if not empty
            if rows:
                all_tables.append({
                    "caption": caption,
                    "headers": headers,
                    "rows": rows,
                    "extraction_method": "confluence_macro"
                })
        
        # Method 3: Grid/Layout-based tables
        for grid_div in soup.find_all('div', class_=lambda c: c and ('grid' in c or 'layout' in c)):
            # Check if this looks like a table structure
            cells = grid_div.find_all('div', class_=lambda c: c and 'cell' in c)
            if len(cells) >= 4:  # At least 4 cells to consider it a table
                rows = []
                current_row = []
                row_idx = -1
                
                for cell in cells:
                    # Try to detect row changes based on positioning
                    cell_style = cell.get('style', '')
                    if 'clear:' in cell_style or 'clear: ' in cell_style:
                        if current_row:
                            rows.append(current_row)
                            current_row = []
                    
                    current_row.append(cell.get_text().strip())
                    
                    # End of row detection based on cell count or explicit markup
                    if len(current_row) >= 3 or 'clear:' in cell_style:
                        rows.append(current_row)
                        current_row = []
                
                # Add any remaining cells
                if current_row:
                    rows.append(current_row)
                
                # Normalize row lengths
                max_cols = max(len(row) for row in rows) if rows else 0
                for i, row in enumerate(rows):
                    if len(row) < max_cols:
                        rows[i] = row + [''] * (max_cols - len(row))
                
                # Add to tables if we have legitimate content
                if rows and max_cols >= 2:
                    # Try to determine if first row is header
                    headers = []
                    if rows:
                        first_is_header = False
                        
                        # Check if first row has different formatting
                        first_row_cells = cells[:max_cols]
                        other_cells = cells[max_cols:]
                        
                        if first_row_cells and other_cells:
                            first_classes = ''.join(str(cell.get('class', '')) for cell in first_row_cells)
                            other_classes = ''.join(str(cell.get('class', '')) for cell in other_cells[:max_cols])
                            
                            if 'header' in first_classes or 'heading' in first_classes or first_classes != other_classes:
                                first_is_header = True
                        
                        if first_is_header:
                            headers = rows[0]
                            rows = rows[1:]
                    
                    all_tables.append({
                        "caption": f"Grid Table",
                        "headers": headers,
                        "rows": rows,
                        "extraction_method": "grid_layout"
                    })
        
        return all_tables
    
    @staticmethod
    def extract_images(soup, base_url):
        """
        Extract all images with their context and descriptions.
        Returns list of dictionaries with image information.
        """
        images = []
        
        for img in soup.find_all('img'):
            # Get basic attributes
            src = img.get('src', '')
            alt = img.get('alt', '')
            title = img.get('title', '')
            
            # Skip icons, emojis and tiny images
            if 'icon' in src.lower() or 'emoji' in src.lower() or 'logo' in src.lower():
                continue
                
            # Try width/height attributes
            width = img.get('width', '')
            height = img.get('height', '')
            
            # Skip very small images (likely icons)
            if (width and int(width.replace('px', '')) < 50) or (height and int(height.replace('px', '')) < 50):
                continue
            
            # Get full image URL
            if src and not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            
            # Get context for image
            context = ComprehensiveContentExtractor._get_image_context(img, soup)
            
            # Add to images list
            images.append({
                "src": src,
                "alt": alt or context.get('alt', ''),
                "title": title or context.get('title', ''),
                "caption": context.get('caption', ''),
                "surrounding_text": context.get('surrounding_text', ''),
                "section": context.get('section', '')
            })
        
        return images
    
    @staticmethod
    def _get_image_context(img, soup):
        """Get comprehensive context for an image."""
        context = {
            'alt': img.get('alt', ''),
            'title': img.get('title', ''),
            'caption': '',
            'surrounding_text': '',
            'section': ''
        }
        
        # Find caption from figure/figcaption
        figure = img.find_parent('figure')
        if figure:
            figcaption = figure.find('figcaption')
            if figcaption:
                context['caption'] = figcaption.get_text().strip()
        
        # If no figcaption, look for nearby div with caption class
        if not context['caption']:
            img_parent = img.parent
            next_sibling = img_parent.next_sibling
            if next_sibling and isinstance(next_sibling, Tag):
                if 'caption' in next_sibling.get('class', []):
                    context['caption'] = next_sibling.get_text().strip()
        
        # Find surrounding text
        surrounding = []
        
        # Previous paragraph
        prev_p = img.find_previous('p')
        if prev_p:
            surrounding.append(prev_p.get_text().strip())
        
        # Next paragraph
        next_p = img.find_next('p')
        if next_p:
            surrounding.append(next_p.get_text().strip())
        
        context['surrounding_text'] = ' '.join(surrounding)
        
        # Find section/heading
        section = img.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if section:
            context['section'] = section.get_text().strip()
        
        return context
    
    @staticmethod
    def extract_links(soup, base_url):
        """
        Extract all relevant links from the page.
        Returns list of dictionaries with link information.
        """
        links = []
        
        for a in soup.find_all('a'):
            href = a.get('href', '')
            text = a.get_text().strip()
            
            # Skip empty links
            if not href or not text:
                continue
            
            # Skip anchors
            if href.startswith('#'):
                continue
            
            # Skip external links (not in the same Confluence instance)
            parsed_base = urlparse(base_url)
            parsed_href = urlparse(href)
            
            is_internal = False
            if not parsed_href.netloc:  # Relative URL
                is_internal = True
            elif parsed_href.netloc == parsed_base.netloc:  # Same domain
                is_internal = True
            
            # Get full URL
            full_url = urljoin(base_url, href)
            
            # Try to determine if it's a Confluence page link
            is_confluence_page = False
            if is_internal:
                if '/pages/' in href or '/display/' in href or '/spaces/' in href:
                    is_confluence_page = True
                
                # Check for viewpage.action
                if 'viewpage.action' in href:
                    is_confluence_page = True
            
            # Extract page ID from URL if possible
            page_id = None
            if is_confluence_page:
                # Try to parse page ID from URL
                if 'pageId=' in href:
                    page_id = parse_qs(urlparse(href).query).get('pageId', [None])[0]
            
            # Get context for link
            link_context = ""
            parent_p = a.find_parent('p')
            if parent_p:
                link_context = parent_p.get_text().strip()
            
            links.append({
                "url": full_url,
                "text": text,
                "is_internal": is_internal,
                "is_confluence_page": is_confluence_page,
                "page_id": page_id,
                "context": link_context
            })
        
        return links
    
    @staticmethod
    def extract_definitions(soup):
        """
        Extract definitions from the page (especially important for acronyms like COPPER).
        Returns list of dictionaries with term and definition.
        """
        definitions = []
        
        # Method 1: Formal definition lists
        for dl in soup.find_all('dl'):
            for dt in dl.find_all('dt'):
                dd = dt.find_next('dd')
                if dd:
                    definitions.append({
                        "term": dt.get_text().strip(),
                        "definition": dd.get_text().strip(),
                        "type": "definition_list"
                    })
        
        # Method 2: Look for typical definition patterns in text
        # First pass: Look for "X stands for Y" or "X is an acronym for Y" patterns
        for p in soup.find_all(['p', 'div']):
            text = p.get_text().strip()
            
            # Look for "COPPER stands for..." or "COPPER is an acronym for..."
            for pattern in [
                r'([A-Z][A-Za-z0-9_-]+)\s+stands\s+for\s+([^\.]+)',
                r'([A-Z][A-Za-z0-9_-]+)\s+is\s+an\s+acronym\s+for\s+([^\.]+)',
                r'([A-Z][A-Za-z0-9_-]+)\s+is\s+short\s+for\s+([^\.]+)',
                r'full\s+form\s+of\s+([A-Z][A-Za-z0-9_-]+)\s+is\s+([^\.]+)',
                r'term\s+([A-Z][A-Za-z0-9_-]+)\s+refers\s+to\s+([^\.]+)'
            ]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        term, definition = match
                        # Skip if term is too common
                        if term.lower() not in ['a', 'an', 'the', 'is', 'are', 'were', 'was']:
                            definitions.append({
                                "term": term.strip(),
                                "definition": definition.strip(),
                                "type": "pattern_match",
                                "context": text
                            })
        
        # Method 3: Look for formatting patterns (e.g., bold term followed by definition)
        for bold in soup.find_all(['b', 'strong']):
            term = bold.get_text().strip()
            
            # Skip common formatting
            if term.lower() in ['note', 'warning', 'important', 'example']:
                continue
                
            # Skip if too short or too long
            if len(term) < 2 or len(term) > 50:
                continue
            
            # Get parent paragraph
            parent = bold.find_parent(['p', 'li', 'div'])
            if parent:
                full_text = parent.get_text().strip()
                
                # Get text after the bold term
                term_index = full_text.find(term)
                if term_index >= 0:
                    after_term = full_text[term_index + len(term):].strip()
                    
                    # Check for typical definition separators
                    for separator in [':', '-', '–', '—', '=']:
                        if separator in after_term[:10]:
                            definition = after_term.split(separator, 1)[1].strip()
                            if definition and len(definition) > 5:
                                definitions.append({
                                    "term": term,
                                    "definition": definition,
                                    "type": "formatting_pattern",
                                    "context": full_text
                                })
                                break
        
        # Method 4: Tables that look like term-definition pairs
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            
            # Skip tables with too many columns (unlikely to be definition tables)
            first_row = rows[0] if rows else None
            if not first_row:
                continue
                
            cells = first_row.find_all(['td', 'th'])
            if len(cells) != 2:
                continue
            
            # Check if this looks like a term-definition table
            header_cells = first_row.find_all('th')
            header_text = [cell.get_text().strip().lower() for cell in header_cells]
            
            is_definition_table = False
            if len(header_text) == 2:
                first_col, second_col = header_text
                if any(term in first_col for term in ['term', 'name', 'acronym', 'abbreviation']):
                    if any(term in second_col for term in ['definition', 'description', 'meaning']):
                        is_definition_table = True
            
            if is_definition_table or len(rows) > 1:  # Either detected or has multiple rows
                for row in rows[1:] if is_definition_table else rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) == 2:
                        term = cells[0].get_text().strip()
                        definition = cells[1].get_text().strip()
                        
                        if term and definition and len(term) <= 50:
                            definitions.append({
                                "term": term,
                                "definition": definition,
                                "type": "table_definition",
                                "table_caption": table.find('caption').get_text().strip() if table.find('caption') else ""
                            })
        
        return definitions
    
    @staticmethod
    def extract_structured_sections(soup):
        """
        Extract structured content like notes, warnings, info panels.
        Returns list of dictionaries with type and content.
        """
        sections = []
        
        # Method 1: Standard Confluence macros
        for div in soup.find_all('div', class_=lambda c: c and any(macro in c for macro in ['panel', 'note', 'warning', 'info', 'tip', 'aui-message'])):
            section_type = 'note'
            for cls in div.get('class', []):
                if cls in ['note', 'warning', 'info', 'tip', 'error', 'success']:
                    section_type = cls
                    break
            
            # Get title if available
            title = ""
            title_elem = div.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b', '.aui-message-heading'])
            if title_elem:
                title = title_elem.get_text().strip()
                
                # Remove title element to avoid duplication
                title_elem.extract()
            
            # Get content
            content = div.get_text().strip()
            
            sections.append({
                "type": section_type.upper(),
                "title": title,
                "content": content
            })
        
        # Method 2: Other formatted sections
        for div in soup.find_all('div'):
            # Look for divs with inline styling that might indicate special sections
            style = div.get('style', '')
            
            if style and ('background' in style or 'border' in style):
                # This might be a formatted section
                
                # Get title if available
                title = ""
                title_elem = div.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b'])
                if title_elem:
                    title = title_elem.get_text().strip()
                
                # Get content
                content = div.get_text().strip()
                
                if title and content and title != content:
                    sections.append({
                        "type": "FORMATTED_SECTION",
                        "title": title,
                        "content": content,
                        "style": style
                    })
        
        return sections
    
    @staticmethod
    def extract_code_blocks(soup):
        """
        Extract code blocks from the page.
        Returns list of dictionaries with code content and language.
        """
        code_blocks = []
        
        # Method 1: Standard code blocks
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                # Try to determine language
                language = ""
                if code.get('class'):
                    for cls in code.get('class'):
                        if cls.startswith('language-'):
                            language = cls.replace('language-', '')
                            break
                
                code_blocks.append({
                    "language": language,
                    "content": code.get_text()
                })
            else:
                # Pre without code tag
                code_blocks.append({
                    "language": "",
                    "content": pre.get_text()
                })
        
        # Method 2: Confluence code macros
        for div in soup.find_all('div', class_=lambda c: c and 'code-block' in c):
            # Try to determine language
            language = ""
            if div.get('class'):
                for cls in div.get('class'):
                    if cls.startswith('language-'):
                        language = cls.replace('language-', '')
                        break
            
            code_blocks.append({
                "language": language,
                "content": div.get_text(),
                "type": "confluence_macro"
            })
        
        return code_blocks
    
    @staticmethod
    def extract_all(html_content, base_url):
        """
        Extract all content from HTML.
        Returns comprehensive structured representation of the page content.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Extract all content types
            extracted = {
                "title": ComprehensiveContentExtractor.extract_title(soup),
                "headings": ComprehensiveContentExtractor.extract_headings(soup),
                "tables": ComprehensiveContentExtractor.extract_tables(soup),
                "images": ComprehensiveContentExtractor.extract_images(soup, base_url),
                "links": ComprehensiveContentExtractor.extract_links(soup, base_url),
                "definitions": ComprehensiveContentExtractor.extract_definitions(soup),
                "structured_sections": ComprehensiveContentExtractor.extract_structured_sections(soup),
                "code_blocks": ComprehensiveContentExtractor.extract_code_blocks(soup),
                "raw_text": soup.get_text().strip()
            }
            
            return extracted
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return {
                "title": "Error",
                "headings": [],
                "tables": [],
                "images": [],
                "links": [],
                "definitions": [],
                "structured_sections": [],
                "code_blocks": [],
                "raw_text": f"Error extracting content: {str(e)}"
            }
    
    @staticmethod
    def format_table_for_text(table):
        """Format a table as text for inclusion in context."""
        lines = []
        
        # Add caption
        if table["caption"]:
            lines.append(f"TABLE: {table['caption']}")
        else:
            lines.append("TABLE:")
        
        # Get column widths for alignment
        col_widths = []
        
        # Check headers
        if table["headers"]:
            col_widths = [len(str(h)) for h in table["headers"]]
        
        # Update widths based on data
        for row in table["rows"]:
            for i, cell in enumerate(row):
                if i >= len(col_widths):
                    col_widths.append(0)
                col_widths[i] = max(col_widths[i], min(len(str(cell)), a30))  # Cap width at 30
        
        # Format with headers if available
        if table["headers"]:
            # Ensure all col_widths are initialized
            while len(col_widths) < len(table["headers"]):
                col_widths.append(0)
            
            # Format header row
            header_cells = []
            for i, header in enumerate(table["headers"]):
                header_cells.append(str(header).ljust(col_widths[i]))
            
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        
        # Format data rows
        for row in table["rows"]:
            row_cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    row_cells.append(str(cell).ljust(col_widths[i]))
                else:
                    row_cells.append(str(cell))
            
            lines.append("| " + " | ".join(row_cells) + " |")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_for_context(extracted):
        """Format all extracted content into text for the context."""
        parts = []
        
        # Title
        if extracted["title"]:
            parts.append(f"# {extracted['title']}")
        
        # Definitions (especially important for acronyms like COPPER)
        if extracted["definitions"]:
            definitions_part = ["## Definitions"]
            for definition in extracted["definitions"]:
                definitions_part.append(f"{definition['term']}: {definition['definition']}")
            
            parts.append("\n".join(definitions_part))
        
        # Headings and content
        content_by_section = defaultdict(list)
        
        # Organize content by sections
        
        # Tables
        for table in extracted["tables"]:
            # Try to find closest heading
            section = "General"
            for heading in extracted["headings"]:
                if table["caption"] and heading["text"] in table["caption"]:
                    section = heading["text"]
                    break
            
            content_by_section[section].append({
                "type": "table",
                "content": ComprehensiveContentExtractor.format_table_for_text(table)
            })
        
        # Images
        for image in extracted["images"]:
            section = image["section"] or "General"
            
            image_text = ["IMAGE:"]
            if image["alt"]:
                image_text.append(f"Alt text: {image['alt']}")
            if image["caption"]:
                image_text.append(f"Caption: {image['caption']}")
            if image["surrounding_text"]:
                image_text.append(f"Context: {image['surrounding_text']}")
            
            content_by_section[section].append({
                "type": "image",
                "content": "\n".join(image_text)
            })
        
        # Structured sections
        for section in extracted["structured_sections"]:
            heading = section["title"] or "General"
            
            section_text = [f"[{section['type']}]"]
            if section["title"]:
                section_text.append(f"Title: {section['title']}")
            section_text.append(section["content"])
            
            content_by_section[heading].append({
                "type": "structured_section",
                "content": "\n".join(section_text)
            })
        
        # Code blocks
        for code_block in extracted["code_blocks"]:
            lang = f"{code_block['language']}" if code_block["language"] else ""
            
            content_by_section["Code Examples"].append({
                "type": "code",
                "content": f"```{lang}\n{code_block['content']}\n```"
            })
        
        # Add raw text paragraphs
        paragraphs = re.split(r'\n{2,}', extracted["raw_text"])
        
        current_section = "General"
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this is a heading
            is_heading = False
            for heading in extracted["headings"]:
                if heading["text"] == paragraph:
                    current_section = heading["text"]
                    is_heading = True
                    break
            
            if not is_heading:
                # Skip paragraphs that are already covered in tables, images, etc.
                skip = False
                for content_list in content_by_section.values():
                    for content_item in content_list:
                        if paragraph in content_item["content"]:
                            skip = True
                            break
                    if skip:
                        break
                
                if not skip:
                    content_by_section[current_section].append({
                        "type": "text",
                        "content": paragraph
                    })
        
        # Format by section
        for heading, contents in content_by_section.items():
            # Skip sections with no content
            if not contents:
                continue
            
            section_part = [f"## {heading}"]
            
            for content_item in contents:
                section_part.append(content_item["content"])
            
            parts.append("\n\n".join(section_part))
        
        # Add relevant links
        if extracted["links"]:
            links_part = ["## Related Pages"]
            for link in extracted["links"]:
                if link["is_confluence_page"]:
                    links_part.append(f"* [{link['text']}]({link['url']})")
            
            if len(links_part) > 1:  # Only add if we have links
                parts.append("\n".join(links_part))
        
        return "\n\n".join(parts)


class EnhancedConfluenceClient:
    """Enhanced client for interacting with Confluence API."""
    
    def __init__(self, base_url, username, api_token):
        """Initialize the client."""
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
        self.session.verify = False  # SSL verification disabled as requested
        
        # Thread safety
        self.lock = threading.Lock()
        
        # For following links
        self.processed_links = set()
    
    def _make_request(self, url, method="GET", params=None, data=None, retries=MAX_RETRIES):
        """Make an API request with retries and caching."""
        # Generate cache key
        cache_key = f"{method}_{url}_{json.dumps(params) if params else ''}_{json.dumps(data) if data else ''}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_path = os.path.join(RAW_CONTENT_DIR, f"api_{cache_key}.json")
        
        # Check cache for GET requests
        if method.upper() == "GET" and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read from cache: {str(e)}")
        
        # Make the request
        for attempt in range(retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                if not response.text.strip():
                    return {}
                
                result = response.json()
                
                # Cache GET responses
                if method.upper() == "GET":
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        logger.warning(f"Failed to write to cache: {str(e)}")
                
                return result
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries + 1}): {str(e)}")
                if attempt < retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Request failed after {retries + 1} attempts: {url}")
                    return None
    
    def test_connection(self):
        """Test the connection to Confluence."""
        logger.info("Testing connection to Confluence...")
        response = self._make_request(f"{self.api_url}/space", params={"limit": 1})
        return response is not None
    
    def get_all_pages_in_space(self, space_key):
        """Get all pages in a space."""
        logger.info(f"Getting all pages in space: {space_key}")
        
        # Check for cached page index
        if os.path.exists(PAGE_INDEX_FILE):
            try:
                with open(PAGE_INDEX_FILE, 'rb') as f:
                    pages = pickle.load(f)
                    logger.info(f"Loaded {len(pages)} pages from cache")
                    return pages
            except Exception as e:
                logger.warning(f"Failed to load page index from cache: {str(e)}")
        
        all_pages = []
        start = 0
        limit = 100  # Max allowed by Confluence
        
        while True:
            logger.info(f"Fetching pages: start={start}, limit={limit}")
            
            response = self._make_request(
                f"{self.api_url}/content",
                params={
                    "spaceKey": space_key,
                    "limit": limit,
                    "start": start,
                    "expand": "metadata.labels,history.lastUpdated",
                    "status": "current"
                }
            )
            
            if not response or not response.get("results"):
                break
            
            results = response["results"]
            all_pages.extend(results)
            
            if len(results) < limit:
                break
                
            start += len(results)
            time.sleep(0.1)  # Avoid rate limiting
        
        logger.info(f"Found {len(all_pages)} pages in space {space_key}")
        
        # Save to cache
        try:
            with open(PAGE_INDEX_FILE, 'wb') as f:
                pickle.dump(all_pages, f)
                logger.info(f"Saved {len(all_pages)} pages to cache")
        except Exception as e:
            logger.warning(f"Failed to save page index to cache: {str(e)}")
        
        return all_pages
    
    def get_page_with_content(self, page_id):
        """Get a page with its content."""
        logger.info(f"Getting page content: {page_id}")
        return self._make_request(
            f"{self.api_url}/content/{page_id}",
            params={
                "expand": "body.storage,metadata.labels,history.lastUpdated,children.attachment"
            }
        )
    
    def get_page_content(self, page_id):
        """Get page content and extract all information."""
        # Check cache
        cache_path = os.path.join(PROCESSED_CONTENT_DIR, f"{page_id}.pickle")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to read processed content from cache: {str(e)}")
        
        # Get page content
        page_data = self.get_page_with_content(page_id)
        if not page_data or "body" not in page_data:
            logger.warning(f"Failed to get content for page: {page_id}")
            return None
        
        # Extract HTML content
        html_content = page_data["body"]["storage"]["value"]
        if not html_content:
            return None
        
        # Get page URL
        page_url = f"{self.base_url}/pages/viewpage.action?pageId={page_id}"
        
        # Extract all content
        extracted = ComprehensiveContentExtractor.extract_all(html_content, page_url)
        
        # Add metadata
        extracted["page_id"] = page_id
        extracted["url"] = page_url
        extracted["space_key"] = page_data.get("space", {}).get("key", "")
        extracted["title"] = page_data.get("title", "")
        extracted["labels"] = [label.get("name") for label in page_data.get("metadata", {}).get("labels", {}).get("results", [])]
        extracted["last_updated"] = page_data.get("history", {}).get("lastUpdated", {}).get("when", "")
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(extracted, f)
        except Exception as e:
            logger.warning(f"Failed to write processed content to cache: {str(e)}")
        
        return extracted
    
    def get_content_from_link(self, link, depth=0):
        """
        Get content from a link, following Confluence page links.
        Returns the extracted content or None if not applicable.
        """
        if depth > MAX_LINK_DEPTH:
            return None
        
        if not link.get("is_confluence_page"):
            return None
        
        # Skip already processed links
        url = link.get("url")
        if url in self.processed_links:
            return None
        
        with self.lock:
            self.processed_links.add(url)
        
        # If we have a page ID, use that
        page_id = link.get("page_id")
        if page_id:
            return self.get_page_content(page_id)
        
        # Otherwise, try to fetch by URL
        try:
            # Extract page ID from URL if possible
            parsed_url = urlparse(url)
            if 'pageId=' in parsed_url.query:
                page_id = parse_qs(parsed_url.query).get('pageId', [None])[0]
                if page_id:
                    return self.get_page_content(page_id)
            
            # Fallback: direct HTTP request
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_path = os.path.join(REFERENCED_CONTENT_DIR, f"{cache_key}.pickle")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
            
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            html_content = response.text
            extracted = ComprehensiveContentExtractor.extract_all(html_content, url)
            
            # Add basic metadata
            extracted["url"] = url
            extracted["is_referenced"] = True
            
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(extracted, f)
            except Exception:
                pass
            
            return extracted
        except Exception as e:
            logger.warning(f"Failed to get content from link {url}: {str(e)}")
            return None
    
    def follow_links(self, page_content, max_links=MAX_LINKS_PER_PAGE):
        """
        Follow relevant links from a page to gather more information.
        Returns a list of content from followed links.
        """
        if not page_content or "links" not in page_content:
            return []
        
        relevant_links = []
        
        # Find most relevant links
        for link in page_content["links"]:
            if link.get("is_confluence_page"):
                # Skip external links
                if not link.get("is_internal"):
                    continue
                
                # Calculate relevance based on context
                relevance = 0
                
                # Links with COPPER in text or context
                if "COPPER" in link["text"] or "COPPER" in link.get("context", ""):
                    relevance += 3
                
                # Links with API in text or context
                if "API" in link["text"] or "API" in link.get("context", ""):
                    relevance += 2
                
                # Links with database terms
                db_terms = ["database", "table", "view", "schema", "column", "field"]
                for term in db_terms:
                    if term in link["text"].lower() or term in link.get("context", "").lower():
                        relevance += 1
                
                # Add to relevant links if score > 0
                if relevance > 0:
                    relevant_links.append((link, relevance))
        
        # Sort by relevance and limit
        relevant_links.sort(key=lambda x: x[1], reverse=True)
        relevant_links = relevant_links[:max_links]
        
        # Follow links in parallel
        referenced_content = []
        
        with ThreadPoolExecutor(max_workers=min(len(relevant_links), 3)) as executor:
            futures = []
            for link, _ in relevant_links:
                futures.append(executor.submit(self.get_content_from_link, link))
            
            for future in futures:
                try:
                    content = future.result()
                    if content:
                        referenced_content.append(content)
                except Exception as e:
                    logger.error(f"Error following link: {str(e)}")
        
        return referenced_content
    
    def search_content(self, query, space_key=None, limit=MAX_RESULTS):
        """
        Search for content using CQL.
        Returns search results ordered by relevance.
        """
        logger.info(f"Searching for content: {query}")
        
        # Extract terms from query
        terms = re.findall(r'\b\w{3,}\b', query.lower())
        if not terms:
            return []
        
        # Build CQL query
        space_clause = f'space = "{space_key}" AND ' if space_key else ''
        
        # Use OR between terms for broader results
        term_clauses = []
        for term in terms:
            # Higher weight for title matches
            term_clauses.append(f'title ~ "{term}"^3')
            # Standard weight for content matches
            term_clauses.append(f'text ~ "{term}"')
        
        cql = f'{space_clause}({" OR ".join(term_clauses)})'
        
        # Make the search request
        response = self._make_request(
            f"{self.api_url}/content/search",
            params={"cql": cql, "limit": limit, "expand": "space"}
        )
        
        if not response or "results" not in response:
            return []
        
        results = response["results"]
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "title": result["title"],
                "url": f"{self.base_url}/pages/viewpage.action?pageId={result['id']}",
                "space_key": result.get("space", {}).get("key", ""),
                "type": result.get("type", "")
            })
        
        return formatted_results


class IntelligentContentManager:
    """Manages content extraction, caching, and relevance assessment."""
    
    def __init__(self, confluence_client):
        """Initialize with a Confluence client."""
        self.confluence = confluence_client
        self.extractor = ComprehensiveContentExtractor
        
        # In-memory caches for faster access
        self.page_cache = {}
        self.reference_cache = {}
        
        # Track references between pages
        self.reference_graph = defaultdict(set)
    
    def process_page(self, page_id, follow_links=True):
        """
        Process a page and all its linked content.
        Returns the complete extracted content.
        """
        # Check in-memory cache
        if page_id in self.page_cache:
            return self.page_cache[page_id]
        
        # Get page content
        content = self.confluence.get_page_content(page_id)
        if not content:
            return None
        
        # Cache the content
        self.page_cache[page_id] = content
        
        # Follow links if requested
        referenced_content = []
        if follow_links:
            referenced_content = self.confluence.follow_links(content)
            
            # Update reference graph
            for ref in referenced_content:
                ref_id = ref.get("page_id")
                if ref_id:
                    self.reference_graph[page_id].add(ref_id)
                    self.reference_cache[ref_id] = ref
        
        return content
    
    def assess_relevance(self, content, query):
        """
        Assess the relevance of content to a query.
        Returns a score between 0.0 and 1.0.
        """
        if not content:
            return 0.0
        
        # Extract query terms
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        # Calculate relevance score
        score = 0.0
        
        # Title match (highest weight)
        title = content.get("title", "").lower()
        title_matches = sum(1 for term in query_terms if term in title)
        title_score = title_matches / len(query_terms) if query_terms else 0
        score += title_score * 0.4
        
        # Check if query is about "what is COPPER" or similar
        is_definition_query = any(p in query_lower for p in ["what is", "definition", "full form", "meaning of", "stands for"])
        if is_definition_query and "COPPER" in query.upper():
            # Check definitions for relevance
            for definition in content.get("definitions", []):
                if "COPPER" in definition.get("term", ""):
                    score += 0.4
                    break
        
        # Check if query is about API endpoints
        is_api_query = any(p in query_lower for p in ["api", "endpoint", "rest", "http", "url"])
        if is_api_query:
            # Check for API references in tables and text
            has_api_tables = any("API" in table.get("caption", "") for table in content.get("tables", []))
            if has_api_tables:
                score += 0.3
            
            # Check for API or endpoint in headings
            api_headings = any("API" in heading.get("text", "") or "endpoint" in heading.get("text", "").lower() for heading in content.get("headings", []))
            if api_headings:
                score += 0.2
        
        # Raw text match for all query terms
        raw_text = content.get("raw_text", "").lower()
        for term in query_terms:
            if term in raw_text:
                score += 0.05
        
        # Special handling for tables
        tables = content.get("tables", [])
        if tables:
            table_score = 0.0
            for table in tables:
                # Check table caption for query terms
                caption = table.get("caption", "").lower()
                caption_matches = sum(1 for term in query_terms if term in caption)
                if caption_matches > 0:
                    table_score += 0.1
                
                # Check table content for query terms
                table_text = "\n".join(" ".join(str(cell) for cell in row) for row in table.get("rows", []))
                table_text = table_text.lower()
                table_matches = sum(1 for term in query_terms if term in table_text)
                if table_matches > 0:
                    table_score += 0.1
            
            score += min(0.3, table_score)  # Cap table score at 0.3
        
        # Normalize score to be between 0 and 1
        return min(1.0, score)
    
    def get_relevant_content(self, query, page_ids, threshold=0.6):
        """
        Get content relevant to the query from the given pages.
        Returns list of (content, relevance) tuples sorted by relevance.
        """
        relevant_content = []
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=min(len(page_ids), MAX_WORKERS)) as executor:
            futures = []
            for page_id in page_ids:
                futures.append(executor.submit(self.process_page, page_id))
            
            for future in futures:
                try:
                    content = future.result()
                    if content:
                        relevance = self.assess_relevance(content, query)
                        if relevance >= threshold:
                            relevant_content.append((content, relevance))
                except Exception as e:
                    logger.error(f"Error processing page: {str(e)}")
        
        # Sort by relevance
        relevant_content.sort(key=lambda x: x[1], reverse=True)
        
        return relevant_content
    
    def format_for_context(self, content_items):
        """
        Format content items into a context string for Gemini.
        Returns formatted context string.
        """
        if not content_items:
            return "No relevant content found."
        
        context_parts = []
        
        for content, relevance in content_items:
            title = content.get("title", "Untitled")
            url = content.get("url", "")
            
            # Format the content
            formatted = self.extractor.format_for_context(content)
            
            context_parts.append(f"--- PAGE: {title} ---\n{formatted}\nSOURCE: {url}\nRELEVANCE: {relevance:.2f}")
        
        return "\n\n".join(context_parts)


class UltimateAssistant:
    """Ultimate assistant for COPPER knowledge with comprehensive extraction."""
    
    def __init__(self, confluence_url, username, api_token, space_key):
        """Initialize the assistant."""
        self.confluence = EnhancedConfluenceClient(confluence_url, username, api_token)
        self.content_manager = IntelligentContentManager(self.confluence)
        self.space_key = space_key
        self.pages = []
        
        # Search index for quick lookups
        self.page_index = {}  # word -> [page_ids]
        
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
    
    def initialize(self):
        """Initialize the assistant."""
        logger.info("Initializing Ultimate COPPER Assistant...")
        
        # Test connection
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence")
            return False
        
        # Load all pages
        self.pages = self.confluence.get_all_pages_in_space(self.space_key)
        if not self.pages:
            logger.error(f"No pages found in space: {self.space_key}")
            return False
        
        # Build simple index for quick lookups
        logger.info("Building search index...")
        self._build_index()
        
        logger.info("Initialization complete")
        return True
    
    def _build_index(self):
        """Build a simple search index for quick lookups."""
        # Check for existing index
        index_path = os.path.join(CACHE_DIR, "word_index.pickle")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    self.page_index = pickle.load(f)
                logger.info(f"Loaded search index with {len(self.page_index)} terms")
                return
            except Exception as e:
                logger.warning(f"Failed to load search index: {str(e)}")
        
        # Build from titles and labels
        for page in self.pages:
            page_id = page.get("id")
            title = page.get("title", "").lower()
            
            # Index title words
            title_words = re.findall(r'\b\w{3,}\b', title)
            for word in title_words:
                if word not in self.page_index:
                    self.page_index[word] = []
                self.page_index[word].append(page_id)
            
            # Index labels
            labels = [label.get("name", "").lower() for label in 
                     page.get("metadata", {}).get("labels", {}).get("results", [])]
            
            for label in labels:
                label_words = re.findall(r'\b\w{3,}\b', label)
                for word in label_words:
                    if word not in self.page_index:
                        self.page_index[word] = []
                    self.page_index[word].append(page_id)
        
        # Save index
        try:
            with open(index_path, 'wb') as f:
                pickle.dump(self.page_index, f)
            logger.info(f"Saved search index with {len(self.page_index)} terms")
        except Exception as e:
            logger.warning(f"Failed to save search index: {str(e)}")
    
    def _search_index(self, query, max_results=MAX_RESULTS):
        """Search the index for pages matching the query."""
        # Extract query terms
        query_terms = re.findall(r'\b\w{3,}\b', query.lower())
        
        if not query_terms:
            return []
        
        # Count matches for each page
        page_counts = defaultdict(int)
        
        for term in query_terms:
            if term in self.page_index:
                for page_id in self.page_index[term]:
                    page_counts[page_id] += 1
        
        # Sort by match count
        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_results = sorted_pages[:max_results]
        
        return [page_id for page_id, _ in top_results]
    
    def search(self, query):
        """
        Search for content related to the query using multiple methods.
        Returns list of page IDs to check.
        """
        logger.info(f"Searching for: {query}")
        
        # Method 1: Use index search
        indexed_results = self._search_index(query)
        
        # Method 2: Use Confluence's API search
        api_results = self.confluence.search_content(query, self.space_key)
        api_page_ids = [result["id"] for result in api_results]
        
        # Combine results without duplicates
        all_page_ids = list(set(indexed_results + api_page_ids))
        
        # Special handling for "what is COPPER" queries
        if "what is copper" in query.lower() or "copper stands for" in query.lower() or "full form of copper" in query.lower():
            # Look for pages with COPPER in title
            for page in self.pages:
                if "COPPER" in page.get("title", ""):
                    if page["id"] not in all_page_ids:
                        all_page_ids.append(page["id"])
        
        # Special handling for API endpoint queries
        if "api" in query.lower() or "endpoint" in query.lower():
            # Look for pages with API in title
            for page in self.pages:
                if "API" in page.get("title", ""):
                    if page["id"] not in all_page_ids:
                        all_page_ids.append(page["id"])
        
        logger.info(f"Found {len(all_page_ids)} candidate pages")
        
        # If no results, return recently updated pages
        if not all_page_ids:
            logger.warning("No search results, using recent pages")
            
            # Sort by last updated
            recent_pages = sorted(
                self.pages,
                key=lambda p: p.get("history", {}).get("lastUpdated", {}).get("when", "2000-01-01"),
                reverse=True
            )[:10]
            
            all_page_ids = [page["id"] for page in recent_pages]
        
        return all_page_ids
    
    def answer_question(self, question):
        """Answer a question with detailed information from all sources."""
        logger.info(f"Processing question: {question}")
        
        start_time = time.time()
        
        # Step 1: Search for relevant pages
        candidate_page_ids = self.search(question)
        
        if not candidate_page_ids:
            logger.warning("No candidate pages found")
            return "I couldn't find specific information about that in the COPPER documentation. Could you try rephrasing your question?"
        
        # Step 2: Assess relevance and get content
        relevant_content = self.content_manager.get_relevant_content(question, candidate_page_ids)
        
        if not relevant_content:
            logger.warning("No relevant content found")
            return "I found some pages related to your question, but they don't contain specific information about what you're asking. Could you try asking in a different way?"
        
        # Step 3: Format content for context
        context = self.content_manager.format_for_context(relevant_content)
        
        # Step 4: Generate answer
        answer = self._generate_answer(question, context)
        
        end_time = time.time()
        logger.info(f"Question answered in {end_time - start_time:.2f} seconds")
        
        return answer
    
    def _generate_answer(self, question, context):
        """Generate an answer using Gemini."""
        logger.info("Generating answer with Gemini")
        
        try:
            # Create system prompt optimized for COPPER knowledge
            system_prompt = """
            You are the Ultimate COPPER Knowledge Assistant, an expert on the COPPER database system, its views, and REST APIs.
            
            When answering:
            1. Be direct and thorough - answer the specific question first
            2. Include specific details, values, and examples from the documentation
            3. If asked about the full form or meaning of COPPER, provide that information explicitly
            4. If asked about API endpoints, list them with their full details including parameters and responses
            5. Format information from tables clearly to preserve structure
            6. Include captions and context from images when relevant
            7. Use code examples when they help explain a concept
            
            When working with the provided context:
            - Look for exact definitions or explanations in the text
            - Pay special attention to structured content like tables and definitions
            - Consider all pages in the context, not just the first one
            - If there are conflicting answers, explain the differences
            - Synthesize information from multiple sources when appropriate
            
            Remember: You have been provided with comprehensive documentation about COPPER. Use this information
            to give precise, helpful answers with specific details and examples.
            """
            
            # Full prompt with context
            full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
            
            # Generate response
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2048,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            if response.candidates and response.candidates[0].text:
                return response.candidates[0].text.strip()
            else:
                logger.warning("Empty response from Gemini")
                return "I couldn't generate a response based on the information I found. Please try asking in a different way."
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while processing your question. Please try again or ask a different question."


def main():
    """Main entry point for the assistant."""
    print("\n==========================================")
    print("  Ultimate COPPER Knowledge Assistant")
    print("==========================================")
    print("\nInitializing...")
    
    # Check for required environment variables
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN:
        print("Error: Missing Confluence credentials. Please set CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN environment variables.")
        return
    
    # Initialize assistant
    assistant = UltimateAssistant(
        confluence_url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        api_token=CONFLUENCE_API_TOKEN,
        space_key=CONFLUENCE_SPACE
    )
    
    if not assistant.initialize():
        print("Error: Failed to initialize assistant. Please check the logs for details.")
        return
    
    print("✓ Connected to Confluence and loaded page information")
    print(f"✓ Indexed {len(assistant.pages)} pages from the {CONFLUENCE_SPACE} space")
    print(f"✓ Comprehensive content extraction ready")
    print("\nYou can now ask questions about COPPER. The assistant will extract information")
    print("from all available sources including tables, images, and linked pages.")
    print("\nType 'exit' to quit.")
    
    # Main loop
    while True:
        user_input = input("\nQuestion: ").strip()
        
        if user_input.lower() in ('exit', 'quit', 'q'):
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\nSearching and extracting information...")
        
        try:
            start_time = time.time()
            answer = assistant.answer_question(user_input)
            end_time = time.time()
            
            print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
            print("-----------------------------------------------")
            print(answer)
            print("-----------------------------------------------")
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print(f"Sorry, I encountered an error: {str(e)}")


if __name__ == "__main__":
    main()
