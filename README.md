# project
    
import sys
import subprocess
import os
import re
import glob
import json
import time
import logging
import importlib.util
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
import tempfile
from datetime import datetime
import string
import random
import uuid
import pickle
import numpy as np
import warnings

# Silence unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def install_packages():
    """Install or upgrade required packages."""
    print("Checking and installing required packages...\n")
    
    base_packages = [
        'nltk',
        'spacy',
        'scikit-learn',
        'numpy',
        'pandas',
        'python-docx',
        'PyPDF2',
        'reportlab',
        'sumy',
        'rank_bm25',
        'matplotlib',
        'tabulate',
        'rouge',
        'tqdm'
    ]
    
    enhanced_packages = [
        'sentence-transformers',
        'faiss-cpu',
        'transformers',
        'torch',
        'networkx'
    ]
    
    multimodal_packages = [
        'pytesseract',
        'pdf2image',
        'pillow',
        'camelot-py[cv]'
    ]
    
    def check_and_install(package, optional=False):
        """Check if a package is installed and install if needed."""
        package_name = package.split('[')[0]  # Handle packages with extras like camelot-py[cv]
        try:
            importlib.import_module(package_name.replace('-', '_'))
            print(f"  ✓ {package} already installed")
            return True
        except ImportError:
            try:
                if optional:
                    print(f"  → Installing optional package {package}...")
                else:
                    print(f"  → Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"  ✓ {package} installed successfully")
                return True
            except Exception as e:
                if optional:
                    print(f"  ✗ Could not install optional package {package}. Some features will be disabled. Error: {str(e)}")
                else:
                    print(f"  ✗ Failed to install {package}. Error: {str(e)}")
                return False
    
    # Install base packages (required)
    print("Installing core packages:")
    for package in base_packages:
        check_and_install(package)
    
    # Install enhanced packages (recommended)
    print("\nInstalling enhanced packages (recommended):")
    enhanced_available = {}
    for package in enhanced_packages:
        enhanced_available[package] = check_and_install(package, optional=True)
    
    # Install multimodal packages (optional)
    print("\nInstalling multimodal packages (optional):")
    multimodal_available = {}
    for package in multimodal_packages:
        multimodal_available[package] = check_and_install(package, optional=True)
    
    # Return information about installed packages
    return {
        "enhanced": enhanced_available,
        "multimodal": multimodal_available
    }


# Install required packages and get availability information
package_availability = install_packages()

# Now import necessary libraries (some may be conditionally imported based on availability)
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import docx
import PyPDF2
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from tqdm import tqdm

# Download required NLTK data
print("\nDownloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Conditional imports based on package availability
sentence_transformers_available = package_availability["enhanced"].get("sentence-transformers", False)
torch_available = package_availability["enhanced"].get("torch", False)
transformers_available = package_availability["enhanced"].get("transformers", False)
faiss_available = package_availability["enhanced"].get("faiss-cpu", False)
tesseract_available = package_availability["multimodal"].get("pytesseract", False)
camelot_available = package_availability["multimodal"].get("camelot-py[cv]", False)

if sentence_transformers_available:
    from sentence_transformers import SentenceTransformer, util as st_util
    try:
        print("Loading SentenceTransformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {str(e)}")
        sentence_transformers_available = False

if transformers_available and torch_available:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    try:
        print("Loading summarization model...")
        summarization_model_name = "facebook/bart-large-cnn"
        summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
        summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
        summarizer = pipeline("summarization", model=summarization_model, tokenizer=summarization_tokenizer)
        transformers_summarization_available = True
    except Exception as e:
        print(f"Error loading transformers models: {str(e)}")
        transformers_summarization_available = False
else:
    transformers_summarization_available = False

if faiss_available:
    import faiss

if tesseract_available:
    import pytesseract
    from PIL import Image as PILImage
    try:
        import pdf2image
        pdf_to_image_available = True
    except:
        pdf_to_image_available = False

if camelot_available:
    try:
        import camelot
        camelot_working = True
    except:
        camelot_working = False
else:
    camelot_working = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedRAGSystem")

# Download required spaCy models
print("\nChecking spaCy models...")
import spacy
try:
    # Try to load a model with word vectors if available
    try:
        nlp = spacy.load("en_core_web_md")
        print("  ✓ Using spaCy model with word vectors (en_core_web_md)")
        has_vectors = True
    except:
        print("  → Downloading spaCy model with word vectors...")
        try:
            spacy.cli.download("en_core_web_md")
            nlp = spacy.load("en_core_web_md")
            print("  ✓ Successfully downloaded and loaded en_core_web_md")
            has_vectors = True
        except:
            print("  ✗ Could not download en_core_web_md. Will use smaller model.")
            try:
                nlp = spacy.load("en_core_web_sm")
                print("  ✓ Using smaller spaCy model (en_core_web_sm)")
                has_vectors = False
            except:
                print("  → Downloading small spaCy model...")
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                print("  ✓ Successfully downloaded and loaded en_core_web_sm")
                has_vectors = False
except Exception as e:
    print(f"  ✗ Error loading spaCy models: {str(e)}")
    sys.exit(1)


class Document:
    """
    Represents a document with metadata and content, allowing for chunking and indexing.
    """
    
    def __init__(self, doc_id, file_path=None, content=None, metadata=None, chunks=None):
        self.id = doc_id
        self.file_path = file_path
        self.content = content
        self.metadata = metadata or {}
        self.chunks = chunks or []
        # Add automatically generated metadata
        if file_path:
            self.metadata["filename"] = os.path.basename(file_path)
            self.metadata["file_extension"] = os.path.splitext(file_path)[1].lower()
        self.metadata["doc_id"] = doc_id
        self.metadata["length"] = len(content) if content else 0
        self.metadata["creation_time"] = datetime.now().isoformat()
    
    def create_chunks(self, chunker):
        """Create chunks from document content using the provided chunker."""
        if not self.content:
            return []
        
        self.chunks = chunker.chunk_document(self)
        return self.chunks
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "content": self.content,
            "metadata": self.metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a Document instance from a dictionary."""
        doc = cls(
            doc_id=data["id"],
            file_path=data.get("file_path"),
            content=data.get("content"),
            metadata=data.get("metadata", {})
        )
        # Reconstruct chunks
        doc.chunks = [DocumentChunk.from_dict(chunk_data, doc) for chunk_data in data.get("chunks", [])]
        return doc
    
    def __str__(self):
        return f"Document(id={self.id}, filename={self.metadata.get('filename')}, chunks={len(self.chunks)})"


class DocumentChunk:
    """
    Represents a chunk of text from a document, with its own metadata, embeddings and context.
    """
    
    def __init__(self, chunk_id, text, document, metadata=None):
        self.id = chunk_id
        self.text = text
        self.document = document  # Reference to parent document
        self.metadata = metadata or {}
        self.metadata["chunk_id"] = chunk_id
        self.metadata["doc_id"] = document.id
        self.metadata["length"] = len(text) if text else 0
        self.embedding = None
    
    def set_embedding(self, embedding):
        """Set the embedding vector for this chunk."""
        self.embedding = embedding
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            # Don't include the document to avoid circularity
            # Don't include embedding as it may not be serializable
        }
    
    @classmethod
    def from_dict(cls, data, document):
        """Create a DocumentChunk instance from a dictionary."""
        return cls(
            chunk_id=data["id"],
            text=data["text"],
            document=document,
            metadata=data.get("metadata", {})
        )
    
    def __str__(self):
        return f"DocumentChunk(id={self.id}, doc_id={self.document.id}, length={len(self.text)})"


class AdaptiveChunker:
    """
    Chunks documents into semantically coherent pieces using adaptive sizing based on content.
    """
    
    def __init__(self, min_chunk_size=100, max_chunk_size=512, chunk_overlap=50):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document):
        """
        Chunk a document into semantically coherent pieces.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        if not document.content:
            return []
        
        # Check if the document has structure (headings, sections, etc.)
        if self._has_structure(document.content):
            return self._chunk_structured_document(document)
        else:
            return self._chunk_unstructured_document(document)
    
    def _has_structure(self, text):
        """Check if the text has structural elements like headings."""
        # Look for heading patterns like Markdown headings or numbered sections
        heading_patterns = [
            r'^\s*#{1,6}\s+.+$',           # Markdown headings
            r'^\s*\d+\.\s+.+$',             # Numbered sections
            r'^\s*[A-Z][A-Za-z ]+:',        # Title-case labels with colon
            r'^\s*[A-Z][A-Z\s]+(?:\s|$)'    # ALL CAPS headings
        ]
        
        lines = text.split('\n')
        heading_count = 0
        
        for line in lines:
            if any(re.match(pattern, line, re.MULTILINE) for pattern in heading_patterns):
                heading_count += 1
        
        # If there are multiple headings, consider it structured
        return heading_count >= 2
    
    def _chunk_structured_document(self, document):
        """Chunk a document that has structural elements like headings."""
        chunks = []
        chunk_id_base = f"{document.id}_chunk_"
        
        # Split text by potential section boundaries
        section_patterns = [
            r'^\s*#{1,6}\s+.+$',           # Markdown headings
            r'^\s*\d+\.\d*\s+.+$',          # Numbered sections and subsections
            r'^\s*[A-Z][A-Za-z ]+:',        # Title-case labels with colon
            r'^\s*[A-Z][A-Z\s]+(?:\s|$)'    # ALL CAPS headings
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
        
        # Split content by section boundaries while keeping the boundaries
        import re
        sections = []
        current_section = []
        lines = document.content.split('\n')
        
        for line in lines:
            if any(re.match(pattern, line, re.MULTILINE) for pattern in section_patterns):
                # Start a new section
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Process each section
        for i, section in enumerate(sections):
            # If section is very small, try to combine with the next section
            if i < len(sections) - 1 and len(section.split()) < self.min_chunk_size:
                sections[i+1] = section + '\n\n' + sections[i+1]
                continue
                
            # If section is too large, split it further
            if len(section.split()) > self.max_chunk_size:
                subsections = self._split_large_text(section)
                for j, subsection in enumerate(subsections):
                    chunk_id = f"{chunk_id_base}{i}_{j}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text=subsection,
                        document=document,
                        metadata={"section_index": i, "subsection_index": j}
                    )
                    chunks.append(chunk)
            else:
                chunk_id = f"{chunk_id_base}{i}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=section,
                    document=document,
                    metadata={"section_index": i}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_unstructured_document(self, document):
        """Chunk a document that lacks clear structural elements."""
        chunks = []
        chunk_id_base = f"{document.id}_chunk_"
        
        # First try to split by paragraphs
        paragraphs = re.split(r'\n\s*\n', document.content)
        
        # If paragraphs are too small or too large, adjust the approach
        if len(paragraphs) <= 1 or max(len(p.split()) for p in paragraphs) > self.max_chunk_size:
            # Use sentence-based splitting
            return self._chunk_by_sentences(document)
        
        # Process paragraphs
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            # If paragraph alone is too large, split it further
            if para_size > self.max_chunk_size:
                # First, add the current chunk if it's not empty
                if current_chunk:
                    chunk_id = f"{chunk_id_base}{chunk_index}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text='\n\n'.join(current_chunk),
                        document=document,
                        metadata={"chunk_index": chunk_index}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0
                
                # Then split the large paragraph and add each piece as a separate chunk
                sentences = nltk.sent_tokenize(para)
                sent_chunks = self._group_sentences(sentences, self.max_chunk_size)
                
                for sent_chunk in sent_chunks:
                    chunk_id = f"{chunk_id_base}{chunk_index}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text=sent_chunk,
                        document=document,
                        metadata={"chunk_index": chunk_index, "from_large_paragraph": True}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            # If adding this paragraph would exceed max size, start a new chunk
            elif current_size + para_size > self.max_chunk_size:
                # If the current chunk is too small, and we can fit part of the paragraph, split the paragraph
                if current_size < self.min_chunk_size and para_size > self.min_chunk_size:
                    sentences = nltk.sent_tokenize(para)
                    
                    # Figure out how many sentences we can add to the current chunk
                    remaining_capacity = self.max_chunk_size - current_size
                    current_sent_count = 0
                    current_sent_size = 0
                    
                    for sent in sentences:
                        sent_size = len(sent.split())
                        if current_sent_size + sent_size <= remaining_capacity:
                            current_chunk.append(sent)
                            current_sent_size += sent_size
                            current_sent_count += 1
                        else:
                            break
                    
                    # Finish the current chunk
                    if current_chunk:
                        chunk_id = f"{chunk_id_base}{chunk_index}"
                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            text='\n\n'.join(current_chunk[:-current_sent_count]) + '\n\n' + ' '.join(current_chunk[-current_sent_count:]),
                            document=document,
                            metadata={"chunk_index": chunk_index}
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Start a new chunk with the remaining sentences
                    remaining_sentences = sentences[current_sent_count:]
                    if remaining_sentences:
                        current_chunk = [' '.join(remaining_sentences)]
                        current_size = sum(len(sent.split()) for sent in remaining_sentences)
                    else:
                        current_chunk = []
                        current_size = 0
                
                # Otherwise, just finish the current chunk and start a new one with this paragraph
                else:
                    if current_chunk:
                        chunk_id = f"{chunk_id_base}{chunk_index}"
                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            text='\n\n'.join(current_chunk),
                            document=document,
                            metadata={"chunk_index": chunk_index}
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    current_chunk = [para]
                    current_size = para_size
            
            # Otherwise, add to the current chunk
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_id = f"{chunk_id_base}{chunk_index}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text='\n\n'.join(current_chunk),
                document=document,
                metadata={"chunk_index": chunk_index}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, document):
        """Chunk a document using sentence boundaries."""
        chunks = []
        chunk_id_base = f"{document.id}_chunk_"
        
        sentences = nltk.sent_tokenize(document.content)
        sentence_chunks = self._group_sentences(sentences, self.max_chunk_size)
        
        for i, chunk_text in enumerate(sentence_chunks):
            chunk_id = f"{chunk_id_base}{i}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                document=document,
                metadata={"chunk_index": i, "chunk_method": "sentence"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _group_sentences(self, sentences, max_size, min_size=None):
        """Group sentences into chunks of appropriate size."""
        if min_size is None:
            min_size = self.min_chunk_size
            
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If a single sentence is too large, include it as its own chunk
            if sentence_size > max_size:
                # First add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Then add the large sentence as its own chunk
                chunks.append(sentence)
                continue
            
            # If adding this sentence would exceed max size, start a new chunk
            if current_size + sentence_size > max_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    # This should not normally happen
                    chunks.append(sentence)
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_large_text(self, text):
        """Split large text into smaller chunks while respecting sentence boundaries."""
        sentences = nltk.sent_tokenize(text)
        return self._group_sentences(sentences, self.max_chunk_size)


class DocumentProcessor:
    """
    Handles document loading, processing, and extraction of text and structure.
    Supports various document formats and multimodal content.
    """
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the document processor."""
        logger.info(f"Initializing DocumentProcessor with path: {knowledge_base_path}")
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store Document objects
        self.multimodal_enabled = tesseract_available and pdf_to_image_available
        self.table_extraction_enabled = camelot_working
        
        # Create chunker for document processing
        self.chunker = AdaptiveChunker()
    
    def load_document(self, file_path: str) -> Optional[Document]:
        """
        Load a single document and process it based on its file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object or None if loading failed
        """
        try:
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            doc_id = f"doc_{str(uuid.uuid4())[:8]}"
            
            # Process based on file type
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    return Document(doc_id=doc_id, file_path=file_path, content=content)
            
            elif file_extension == '.docx':
                return self._process_docx(file_path, doc_id)
            
            elif file_extension == '.pdf':
                return self._process_pdf(file_path, doc_id)
            
            else:
                logger.warning(f"Unsupported file type: {file_extension} for file {file_name}")
                return None
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def load_all_documents(self) -> Dict[str, Document]:
        """
        Load all documents from the knowledge base directory.
        
        Returns:
            Dictionary mapping document IDs to Document objects
        """
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path, exist_ok=True)
            logger.warning(f"Created knowledge base directory: {self.knowledge_base_path}")
            
        all_files = glob.glob(os.path.join(self.knowledge_base_path, '*.*'))
        
        if not all_files:
            logger.warning(f"No files found in knowledge base path: {self.knowledge_base_path}")
            return self.documents
        
        logger.info(f"Found {len(all_files)} files in knowledge base.")
        
        for file_path in tqdm(all_files, desc="Loading documents"):
            doc = self.load_document(file_path)
            if doc:
                self.documents[doc.id] = doc
        
        logger.info(f"Successfully loaded {len(self.documents)} documents")
        
        # Chunk all documents
        self._chunk_all_documents()
        
        return self.documents
    
    def _chunk_all_documents(self):
        """Create chunks for all loaded documents."""
        for doc_id, doc in tqdm(self.documents.items(), desc="Chunking documents"):
            if not doc.chunks:
                doc.create_chunks(self.chunker)
    
    def _process_docx(self, file_path: str, doc_id: str) -> Document:
        """Process a DOCX file."""
        doc = docx.Document(file_path)
        
        # Extract basic text content
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        content = '\n\n'.join(paragraphs)
        
        # Extract metadata
        metadata = {
            "title": doc.core_properties.title or os.path.basename(file_path),
            "author": doc.core_properties.author or "Unknown",
            "created": str(doc.core_properties.created) if doc.core_properties.created else "Unknown",
            "modified": str(doc.core_properties.modified) if doc.core_properties.modified else "Unknown",
            "paragraph_count": len(paragraphs)
        }
        
        # Extract tables if any
        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            
            # Convert table to text representation
            try:
                table_str = tabulate(table_data, headers="firstrow", tablefmt="pipe")
                tables.append(table_str)
            except:
                # Simple fallback if tabulate fails
                tables.append('\n'.join([' | '.join(row) for row in table_data]))
        
        # Add tables to content
        if tables:
            content += "\n\n" + "\n\n".join(tables)
            metadata["table_count"] = len(tables)
        
        return Document(doc_id=doc_id, file_path=file_path, content=content, metadata=metadata)
    
    def _process_pdf(self, file_path: str, doc_id: str) -> Document:
        """Process a PDF file with support for text, tables, and images."""
        # Extract basic text content
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            metadata = {
                "page_count": len(pdf_reader.pages),
                "pdf_info": {}
            }
            
            # Extract PDF metadata if available
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if key.startswith('/'):
                        clean_key = key[1:]  # Remove the leading slash
                    else:
                        clean_key = key
                    metadata["pdf_info"][clean_key] = str(value)
            
            # Extract text content
            content = ""
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    content += page_text + "\n\n"
        
        # Extract tables if enabled
        tables = []
        if self.table_extraction_enabled:
            try:
                # Use camelot to extract tables
                table_dfs = camelot.read_pdf(file_path, pages='all')
                
                for i, table in enumerate(table_dfs):
                    df = table.df
                    # Convert dataframe to markdown table
                    table_str = df.to_markdown(index=False)
                    tables.append(f"Table {i+1}:\n{table_str}")
                
                if tables:
                    content += "\n\n" + "\n\n".join(tables)
                    metadata["table_count"] = len(tables)
            except Exception as e:
                logger.warning(f"Error extracting tables from PDF: {str(e)}")
        
        # Extract text from images if OCR is enabled
        if self.multimodal_enabled:
            try:
                # Convert PDF pages to images
                images = pdf2image.convert_from_path(file_path)
                
                image_texts = []
                for i, img in enumerate(images):
                    # Use OCR to extract text from image
                    img_text = pytesseract.image_to_string(img)
                    
                    # Only add if OCR found meaningful text (more than just noise)
                    if len(img_text.strip()) > 20:  # Arbitrary threshold to filter out noise
                        image_texts.append(f"Image {i+1} text: {img_text}")
                
                if image_texts:
                    content += "\n\n" + "\n\n".join(image_texts)
                    metadata["ocr_processed"] = True
                    metadata["image_count"] = len(images)
            except Exception as e:
                logger.warning(f"Error extracting text from PDF images: {str(e)}")
        
        return Document(doc_id=doc_id, file_path=file_path, content=content, metadata=metadata)


class SemanticIndexer:
    """
    Creates and manages semantic indexes for document chunks, supporting both
    vector-based and lexical search capabilities.
    """
    
    def __init__(self, use_faiss=True):
        """Initialize the semantic indexer."""
        self.chunks = []  # All document chunks
        self.doc_id_to_chunks = {}  # Mapping from doc_id to list of chunk indexes
        self.chunk_id_to_index = {}  # Mapping from chunk_id to index in self.chunks
        
        # Embedding-based search components
        self.use_sentence_transformers = sentence_transformers_available
        self.use_faiss = use_faiss and faiss_available
        
        if self.use_sentence_transformers:
            self.sentence_model = sentence_model
        else:
            # Fallback to TF-IDF
            self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Sparse vector indexes (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # BM25 for lexical search
        self.bm25 = None
        self.tokenized_corpus = None
        
        # Vector search indexes
        self.embeddings = None
        self.faiss_index = None
        
        logger.info(f"Initialized SemanticIndexer with sentence_transformers={self.use_sentence_transformers}, faiss={self.use_faiss}")
    
    def index_documents(self, documents):
        """Build indexes for all chunks in the provided documents."""
        logger.info(f"Indexing {len(documents)} documents")
        
        # Collect all chunks
        all_chunks = []
        for doc_id, doc in documents.items():
            if not doc.chunks:
                logger.warning(f"Document {doc_id} has no chunks, skipping")
                continue
                
            chunk_indices = []
            for chunk in doc.chunks:
                chunk_idx = len(all_chunks)
                all_chunks.append(chunk)
                chunk_indices.append(chunk_idx)
                self.chunk_id_to_index[chunk.id] = chunk_idx
                
            self.doc_id_to_chunks[doc_id] = chunk_indices
        
        self.chunks = all_chunks
        
        if not all_chunks:
            logger.warning("No chunks to index!")
            return
            
        logger.info(f"Found {len(all_chunks)} chunks to index")
        
        # Create text corpus for indexing
        corpus = [chunk.text for chunk in all_chunks]
        
        # Create BM25 index
        self.tokenized_corpus = [self._tokenize(text) for text in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Create TF-IDF index
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Create vector embeddings
        if self.use_sentence_transformers:
            embeddings = self._create_embeddings_with_transformers(corpus)
        else:
            embeddings = self.tfidf_matrix
            
        # Store embeddings in chunks
        for i, chunk in enumerate(all_chunks):
            if self.use_sentence_transformers:
                chunk.embedding = embeddings[i]
            else:
                # For TF-IDF, just store the index (can't easily store sparse matrices in objects)
                chunk.embedding = i
        
        # Create FAISS index if enabled
        if self.use_faiss and self.use_sentence_transformers:
            embedding_dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine on normalized vectors)
            self.faiss_index.add(embeddings)
            self.embeddings = embeddings
            logger.info(f"Created FAISS index with {len(embeddings)} vectors of dimension {embedding_dim}")
        elif self.use_sentence_transformers:
            # Store embeddings directly if FAISS not used
            self.embeddings = embeddings
            logger.info(f"Stored {len(embeddings)} embedding vectors of dimension {embeddings.shape[1]}")
        
        logger.info("Indexing completed successfully")
    
    def _create_embeddings_with_transformers(self, texts):
        """Create embeddings using sentence transformers."""
        logger.info(f"Creating embeddings for {len(texts)} texts using SentenceTransformer")
        
        # Process in batches to avoid memory issues with large corpora
        batch_size = 32
        num_texts = len(texts)
        embeddings_list = []
        
        for i in tqdm(range(0, num_texts, batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:min(i+batch_size, num_texts)]
            batch_embeddings = self.sentence_model.encode(batch_texts)
            embeddings_list.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(embeddings_list)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def _tokenize(self, text):
        """Tokenize text for BM25 search."""
        return [word.lower() for word in word_tokenize(text) if word.lower() not in STOPWORDS]
    
    def hybrid_search(self, query, top_k=10, alpha=0.5):
        """
        Perform hybrid search combining lexical and semantic search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha = weight for lexical search)
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self.chunks:
            logger.warning("No indexed chunks available for search!")
            return []
        
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_k * 2)
        lexical_results = self.lexical_search(query, top_k * 2)
        
        # Combine and normalize scores
        combined_scores = {}
        
        # Get max scores for normalization
        max_semantic = max([score for _, score in semantic_results]) if semantic_results else 1.0
        max_lexical = max([score for _, score in lexical_results]) if lexical_results else 1.0
        
        # Normalize and combine semantic results
        for chunk, score in semantic_results:
            combined_scores[chunk.id] = alpha * (score / max_semantic)
        
        # Normalize and combine lexical results
        for chunk, score in lexical_results:
            if chunk.id in combined_scores:
                combined_scores[chunk.id] += (1 - alpha) * (score / max_lexical)
            else:
                combined_scores[chunk.id] = (1 - alpha) * (score / max_lexical)
        
        # Get chunks by ID
        id_to_chunk = {chunk.id: chunk for chunk in self.chunks}
        
        # Sort and return top_k results
        results = sorted([(id_to_chunk[chunk_id], score) 
                         for chunk_id, score in combined_scores.items()], 
                        key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def semantic_search(self, query, top_k=10):
        """
        Perform semantic search using embeddings.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self.chunks:
            return []
        
        if self.use_sentence_transformers:
            # Encode the query
            query_embedding = self.sentence_model.encode(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
            
            if self.use_faiss:
                # Search using FAISS
                scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), top_k)
                results = [(self.chunks[idx], float(score)) for score, idx in zip(scores[0], indices[0])]
            else:
                # Compute cosine similarity manually
                similarities = np.dot(self.embeddings, query_embedding)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                results = [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]
        else:
            # Fallback to TF-IDF
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def lexical_search(self, query, top_k=10):
        """
        Perform lexical search using BM25.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self.chunks:
            return []
        
        # Tokenize query and get BM25 scores
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top results
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        results = [(self.chunks[idx], float(bm25_scores[idx])) for idx in top_indices if bm25_scores[idx] > 0]
        
        return results


class QueryProcessor:
    """
    Analyzes user queries, expands them, and reformulates them to improve retrieval performance.
    """
    
    def __init__(self):
        """Initialize the query processor."""
        # Initialize NLP resources
        self.nlp = nlp
        self.use_wordnet = True
    
    def process_query(self, query):
        """
        Process a query to improve retrieval performance.
        
        Args:
            query: Original user query
            
        Returns:
            Dict containing original query, expanded query, query analysis, etc.
        """
        # Parse the query
        query_doc = self.nlp(query)
        
        # Extract query intent and key concepts
        intent_analysis = self._analyze_intent(query_doc)
        
        # Perform query expansion
        expanded_query = self._expand_query(query, query_doc)
        
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "intent": intent_analysis["intent"],
            "keywords": intent_analysis["keywords"],
            "entities": intent_analysis["entities"],
            "expected_answer_type": intent_analysis["expected_answer_type"]
        }
    
    def _analyze_intent(self, query_doc):
        """
        Analyze the intent behind a query.
        
        Args:
            query_doc: spaCy Doc object of the query
            
        Returns:
            Dict with query intent analysis
        """
        # Default values
        result = {
            "intent": "general_query",
            "keywords": [],
            "entities": [],
            "expected_answer_type": "factoid"
        }
        
        # Extract keywords (important nouns, verbs, and adjectives)
        for token in query_doc:
            if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop:
                result["keywords"].append({"text": token.text, "lemma": token.lemma_, "pos": token.pos_})
            elif token.pos_ in ('VERB', 'ADJ') and token.is_alpha and len(token.text) > 2 and not token.is_stop:
                result["keywords"].append({"text": token.text, "lemma": token.lemma_, "pos": token.pos_})
        
        # Extract named entities
        for ent in query_doc.ents:
            result["entities"].append({"text": ent.text, "label": ent.label_})
            
        # Determine question type and expected answer type
        first_token = query_doc[0].text.lower() if len(query_doc) > 0 else ""
        
        # Map question words to expected answer types
        question_type_mapping = {
            "what": "definition",
            "who": "person",
            "when": "time",
            "where": "location",
            "why": "reason",
            "how": "process"
        }
        
        if first_token in question_type_mapping:
            result["intent"] = "question"
            result["expected_answer_type"] = question_type_mapping[first_token]
        elif "?" in query_doc.text:
            result["intent"] = "question"
        
        # Check for command/instruction intent
        if any(token.lemma_ in ("explain", "describe", "define", "list", "show", "find") for token in query_doc):
            result["intent"] = "instruction"
        
        return result
    
    def _expand_query(self, query, query_doc):
        """
        Expand a query with related terms and synonyms to improve recall.
        
        Args:
            query: Original query string
            query_doc: spaCy Doc object of the query
            
        Returns:
            Expanded query string
        """
        # Extract key terms for expansion
        key_terms = []
        for token in query_doc:
            if token.pos_ in ('NOUN', 'VERB', 'ADJ') and not token.is_stop:
                key_terms.append(token.lemma_)
        
        if not key_terms:
            return query  # Nothing to expand
        
        # Get synonyms using WordNet if available
        expansion_terms = set()
        
        if self.use_wordnet:
            for term in key_terms:
                # Get WordNet synonyms
                synsets = wordnet.synsets(term)[:2]  # Limit to top 2 synsets to avoid too much expansion
                for synset in synsets:
                    lemmas = synset.lemmas()[:3]  # Limit to top 3 synonyms per synset
                    for lemma in lemmas:
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != term and synonym not in query.lower() and len(synonym) > 3:
                            expansion_terms.add(synonym)
        
        # Add hyponyms and hypernyms for nouns
        noun_terms = [term for term, token in zip(key_terms, query_doc) if token.pos_ in ('NOUN', 'PROPN')]
        for term in noun_terms:
            synsets = wordnet.synsets(term, pos=wordnet.NOUN)[:1]  # Limit to top synset
            for synset in synsets:
                # Add hypernyms (more general terms)
                for hypernym in synset.hypernyms()[:1]:
                    lemmas = hypernym.lemmas()[:1]
                    for lemma in lemmas:
                        hypernym_term = lemma.name().replace('_', ' ')
                        if hypernym_term not in query.lower() and len(hypernym_term) > 3:
                            expansion_terms.add(hypernym_term)
                
                # Add hyponyms (more specific terms)
                for hyponym in synset.hyponyms()[:2]:
                    lemmas = hyponym.lemmas()[:1]
                    for lemma in lemmas:
                        hyponym_term = lemma.name().replace('_', ' ')
                        if hyponym_term not in query.lower() and len(hyponym_term) > 3:
                            expansion_terms.add(hyponym_term)
        
        # Limit the number of expansion terms to avoid query dilution
        expansion_terms = list(expansion_terms)[:5]
        
        if expansion_terms:
            expanded_query = query + " " + " ".join(expansion_terms)
            return expanded_query
        else:
            return query


class AbstractiveSummarizer:
    """
    Generates coherent, abstractive summaries from retrieved document chunks.
    """
    
    def __init__(self):
        """Initialize the summarizer with appropriate models."""
        self.use_transformers = transformers_summarization_available
        
        if not self.use_transformers:
            # Fallback to extractive summarization
            self.language = "english"
            self.stemmer = Stemmer(self.language) if sumy_available else None
            
            if sumy_available:
                self.sumy_stop_words = get_stop_words(self.language)
                self.lexrank = LexRankSummarizer(self.stemmer)
                self.lexrank.stop_words = self.sumy_stop_words
                self.lsa = LsaSummarizer(self.stemmer)
                self.lsa.stop_words = self.sumy_stop_words
    
    def generate_summary(self, chunks, query_info, max_length=500):
        """
        Generate a coherent summary from document chunks that addresses the query.
        
        Args:
            chunks: List of (DocumentChunk, score) tuples, ordered by relevance
            query_info: Query analysis information from QueryProcessor
            max_length: Maximum summary length (words for extractive, tokens for abstractive)
            
        Returns:
            Generated summary string
        """
        if not chunks:
            return "No relevant information found."
        
        # Extract text from chunks, ordered by relevance
        chunk_texts = [chunk.text for chunk, _ in chunks]
        combined_text = "\n\n".join(chunk_texts)
        
        # Use abstractive summarization if available
        if self.use_transformers:
            return self._generate_abstractive_summary(chunk_texts, query_info, max_length)
        else:
            # Fall back to extractive summarization
            return self._generate_extractive_summary(combined_text, query_info, max_length)
    
    def _generate_abstractive_summary(self, texts, query_info, max_tokens=500):
        """Generate an abstractive summary using a transformer model."""
        # Prepare the context with the most relevant chunks
        # Limit total length to avoid exceeding model's max length
        MAX_CONTEXT_LENGTH = 4000  # Conservative limit for most models
        
        context = ""
        for text in texts:
            if len(context.split()) + len(text.split()) <= MAX_CONTEXT_LENGTH:
                context += text + "\n\n"
            else:
                break
        
        # Create a prompt that instructs the model what kind of summary to generate
        query = query_info["original_query"]
        intent = query_info["intent"]
        
        if intent == "question":
            prompt = f"Please answer this question based on the provided information: {query}\n\nInformation:\n{context}\n\nAnswer:"
        else:
            prompt = f"Please provide a comprehensive summary of the following information in response to this query: {query}\n\nInformation:\n{context}\n\nSummary:"
        
        try:
            # Generate summary using BART model
            summary = summarizer(prompt, max_length=max_tokens, min_length=50, do_sample=False)[0]['summary_text']
            
            # Clean up the summary
            summary = summary.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating abstractive summary: {str(e)}")
            # Fall back to extractive summarization
            return self._generate_extractive_summary("\n\n".join(texts), query_info, max_tokens)
    
    def _generate_extractive_summary(self, text, query_info, max_sentences=10):
        """Generate an extractive summary using LexRank or LSA."""
        if not sumy_available:
            # Simple extractive summarization
            sentences = nltk.sent_tokenize(text)
            
            # Use TF-IDF to score sentences
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate query similarity if query is provided
            query = query_info["original_query"]
            query_vec = vectorizer.transform([query])
            
            # Score sentences by similarity to query
            scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            
            # Select top sentences
            top_indices = np.argsort(scores)[-max_sentences:]
            top_indices = sorted(top_indices)  # Sort by position to maintain document flow
            
            selected_sentences = [sentences[i] for i in top_indices]
            summary = " ".join(selected_sentences)
            
            return summary
        
        # Use sumy for summarization
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Check if we have enough sentences
        if len(parser.document.sentences) <= max_sentences:
            return text
        
        # Combine LexRank and LSA for better results
        lexrank_sentences = [str(s) for s in self.lexrank(parser.document, max_sentences)]
        lsa_sentences = [str(s) for s in self.lsa(parser.document, max_sentences // 2)]
        
        # Combine and remove duplicates
        all_sentences = []
        all_sentences.extend(lexrank_sentences)
        
        for sentence in lsa_sentences:
            if sentence not in all_sentences:
                all_sentences.append(sentence)
        
        # Limit to max_sentences
        selected_sentences = all_sentences[:max_sentences]
        
        # Reorder sentences to match their original order in the text
        orig_sentences = nltk.sent_tokenize(text)
        orig_order = {}
        
        for i, sent in enumerate(orig_sentences):
            orig_order[sent] = i
        
        # Sort selected sentences by their original order
        selected_sentences.sort(key=lambda s: orig_order.get(s, float('inf')))
        
        # Combine sentences into paragraphs
        return self._format_sentences_into_paragraphs(selected_sentences)
    
    def _format_sentences_into_paragraphs(self, sentences, max_sentences_per_para=3):
        """Format a list of sentences into coherent paragraphs."""
        if not sentences:
            return ""
            
        paragraphs = []
        current_para = []
        
        for sent in sentences:
            current_para.append(sent)
            
            if len(current_para) >= max_sentences_per_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        
        # Add the last paragraph if not empty
        if current_para:
            paragraphs.append(" ".join(current_para))
        
        return "\n\n".join(paragraphs)


class ConversationManager:
    """
    Manages conversation history, context, and user feedback to improve responses over time.
    """
    
    def __init__(self):
        """Initialize the conversation manager."""
        self.history = []
        self.feedback = {}
        self.session_id = str(uuid.uuid4())
        self.user_preferences = {
            "summary_length": "medium",
            "detail_level": "balanced",
            "include_sources": True
        }
    
    def add_interaction(self, query, response, retrieved_docs, chunks):
        """Add an interaction to the conversation history."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "doc_ids": [doc_id for doc_id in retrieved_docs],
            "chunk_ids": [chunk.id for chunk, _ in chunks] if chunks else []
        }
        
        self.history.append(interaction)
        return len(self.history) - 1  # Return index of this interaction
    
    def add_feedback(self, interaction_id, feedback_type, value):
        """Add user feedback for a specific interaction."""
        if interaction_id < 0 or interaction_id >= len(self.history):
            logger.error(f"Invalid interaction_id: {interaction_id}")
            return False
            
        if interaction_id not in self.feedback:
            self.feedback[interaction_id] = {}
            
        self.feedback[interaction_id][feedback_type] = value
        return True
    
    def get_conversation_context(self, current_query, max_history=3):
        """Get relevant conversation context for the current query."""
        if not self.history:
            return None
            
        # Only consider recent history
        recent_history = self.history[-max_history:]
        
        # Extract previous queries and responses
        context = {
            "recent_queries": [item["query"] for item in recent_history],
            "recent_responses": [item["response"] for item in recent_history],
            "mentioned_docs": set()
        }
        
        for item in recent_history:
            for doc_id in item["doc_ids"]:
                context["mentioned_docs"].add(doc_id)
        
        return context
    
    def update_preferences(self, preferences):
        """Update user preferences."""
        for key, value in preferences.items():
            if key in self.user_preferences:
                self.user_preferences[key] = value
            else:
                logger.warning(f"Unknown preference key: {key}")
        
        return self.user_preferences


class EvaluationFramework:
    """
    Framework for evaluating the RAG system on benchmark datasets and custom test cases.
    """
    
    def __init__(self, rag_system):
        """Initialize the evaluation framework with the RAG system to evaluate."""
        self.rag_system = rag_system
        self.metrics = {
            "retrieval": {},
            "summary": {},
            "overall": {}
        }
        
        # Import ROUGE if available
        try:
            from rouge import Rouge
            self.rouge = Rouge()
            self.rouge_available = True
        except:
            self.rouge_available = False
            logger.warning("ROUGE metrics not available. Install 'rouge' package for summary evaluation.")
    
    def evaluate(self, test_cases):
        """
        Evaluate the RAG system on a set of test cases.
        
        Args:
            test_cases: List of dicts containing queries and expected results
            
        Returns:
            Dict of evaluation metrics
        """
        results = {
            "retrieval_precision": [],
            "retrieval_recall": [],
            "response_time": [],
            "summary_rouge": [] if self.rouge_available else None
        }
        
        for test_case in tqdm(test_cases, desc="Evaluating"):
            query = test_case["query"]
            expected_docs = test_case.get("relevant_docs", [])
            reference_summary = test_case.get("reference_summary")
            
            # Process the query and time it
            start_time = time.time()
            response = self.rag_system.process_query(query)
            elapsed = time.time() - start_time
            
            # Evaluate retrieval
            if expected_docs:
                retrieved_docs = response["retrieved_docs"]
                
                # Calculate precision and recall
                retrieved_set = set(retrieved_docs)
                expected_set = set(expected_docs)
                
                true_positives = len(retrieved_set.intersection(expected_set))
                precision = true_positives / len(retrieved_set) if retrieved_set else 0
                recall = true_positives / len(expected_set) if expected_set else 0
                
                results["retrieval_precision"].append(precision)
                results["retrieval_recall"].append(recall)
            
            # Evaluate summary
            if reference_summary and self.rouge_available and "response" in response:
                try:
                    rouge_scores = self.rouge.get_scores(response["response"], reference_summary)
                    results["summary_rouge"].append(rouge_scores[0])
                except Exception as e:
                    logger.error(f"Error calculating ROUGE scores: {str(e)}")
            
            # Record response time
            results["response_time"].append(elapsed)
        
        # Calculate aggregate metrics
        metrics = {
            "avg_precision": sum(results["retrieval_precision"]) / len(results["retrieval_precision"]) if results["retrieval_precision"] else None,
            "avg_recall": sum(results["retrieval_recall"]) / len(results["retrieval_recall"]) if results["retrieval_recall"] else None,
            "avg_response_time": sum(results["response_time"]) / len(results["response_time"]) if results["response_time"] else None,
        }
        
        # Add F1 score
        if metrics["avg_precision"] and metrics["avg_recall"]:
            precision = metrics["avg_precision"]
            recall = metrics["avg_recall"]
            if precision + recall > 0:
                metrics["f1_score"] = 2 * precision * recall / (precision + recall)
            else:
                metrics["f1_score"] = 0
        
        # Add ROUGE metrics if available
        if self.rouge_available and results["summary_rouge"]:
            avg_rouge_1 = sum(entry['rouge-1']['f'] for entry in results["summary_rouge"]) / len(results["summary_rouge"])
            avg_rouge_2 = sum(entry['rouge-2']['f'] for entry in results["summary_rouge"]) / len(results["summary_rouge"])
            avg_rouge_l = sum(entry['rouge-l']['f'] for entry in results["summary_rouge"]) / len(results["summary_rouge"])
            
            metrics["avg_rouge_1"] = avg_rouge_1
            metrics["avg_rouge_2"] = avg_rouge_2
            metrics["avg_rouge_l"] = avg_rouge_l
        
        # Store metrics for later reference
        self.metrics["retrieval"]["precision"] = metrics["avg_precision"]
        self.metrics["retrieval"]["recall"] = metrics["avg_recall"]
        if "f1_score" in metrics:
            self.metrics["retrieval"]["f1"] = metrics["f1_score"]
        
        if self.rouge_available and "avg_rouge_l" in metrics:
            self.metrics["summary"]["rouge_1"] = metrics["avg_rouge_1"]
            self.metrics["summary"]["rouge_2"] = metrics["avg_rouge_2"]
            self.metrics["summary"]["rouge_l"] = metrics["avg_rouge_l"]
        
        self.metrics["overall"]["response_time"] = metrics["avg_response_time"]
        
        return metrics
    
    def generate_report(self, metrics=None, output_path=None):
        """
        Generate an evaluation report with metrics and visualizations.
        
        Args:
            metrics: Metrics dict (if None, use the last evaluation metrics)
            output_path: Path to save the report (if None, return the report as string)
            
        Returns:
            Report string or path to the saved report
        """
        if metrics is None:
            metrics = self.metrics
        
        # Create the report
        report = []
        report.append("# RAG System Evaluation Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Retrieval Performance")
        
        # Retrieval metrics
        if "precision" in metrics["retrieval"]:
            report.append(f"- Precision: {metrics['retrieval']['precision']:.4f}")
        if "recall" in metrics["retrieval"]:
            report.append(f"- Recall: {metrics['retrieval']['recall']:.4f}")
        if "f1" in metrics["retrieval"]:
            report.append(f"- F1 Score: {metrics['retrieval']['f1']:.4f}")
        
        # Summary metrics
        if metrics["summary"]:
            report.append("\n## Summary Quality")
            if "rouge_1" in metrics["summary"]:
                report.append(f"- ROUGE-1: {metrics['summary']['rouge_1']:.4f}")
            if "rouge_2" in metrics["summary"]:
                report.append(f"- ROUGE-2: {metrics['summary']['rouge_2']:.4f}")
            if "rouge_l" in metrics["summary"]:
                report.append(f"- ROUGE-L: {metrics['summary']['rouge_l']:.4f}")
        
        # Performance metrics
        report.append("\n## System Performance")
        if "response_time" in metrics["overall"]:
            report.append(f"- Average Response Time: {metrics['overall']['response_time']:.4f} seconds")
        
        # Generate visualizations if matplotlib is available
        try:
            if output_path:
                self._generate_visualization(metrics, output_path)
                report.append("\n## Visualizations")
                report.append("See attached charts for visual representation of the metrics.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        # Convert to string or save to file
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            return output_path
        else:
            return report_str
    
    def _generate_visualization(self, metrics, output_path):
        """Generate visualization charts for metrics."""
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Retrieval Metrics
        retrieval_metrics = []
        values = []
        
        if "precision" in metrics["retrieval"]:
            retrieval_metrics.append("Precision")
            values.append(metrics["retrieval"]["precision"])
        if "recall" in metrics["retrieval"]:
            retrieval_metrics.append("Recall")
            values.append(metrics["retrieval"]["recall"])
        if "f1" in metrics["retrieval"]:
            retrieval_metrics.append("F1 Score")
            values.append(metrics["retrieval"]["f1"])
            
        if retrieval_metrics:
            axs[0].bar(retrieval_metrics, values, color='skyblue')
            axs[0].set_title('Retrieval Performance')
            axs[0].set_ylim(0, 1)
            axs[0].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for i, v in enumerate(values):
                axs[0].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # Plot 2: ROUGE Metrics if available
        if metrics["summary"] and "rouge_1" in metrics["summary"]:
            rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            rouge_values = [
                metrics["summary"]["rouge_1"],
                metrics["summary"]["rouge_2"],
                metrics["summary"]["rouge_l"]
            ]
            
            axs[1].bar(rouge_metrics, rouge_values, color='lightgreen')
            axs[1].set_title('Summary Quality (ROUGE Metrics)')
            axs[1].set_ylim(0, 1)
            axs[1].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for i, v in enumerate(rouge_values):
                axs[1].text(i, v + 0.02, f'{v:.2f}', ha='center')
        else:
            # If ROUGE not available, plot response time
            axs[1].bar(['Response Time (seconds)'], [metrics["overall"]["response_time"]], color='salmon')
            axs[1].set_title('System Performance')
            axs[1].grid(axis='y', linestyle='--', alpha=0.7)
            axs[1].text(0, metrics["overall"]["response_time"] + 0.1, f'{metrics["overall"]["response_time"]:.2f}s', ha='center')
        
        plt.tight_layout()
        
        # Save to file
        chart_path = os.path.splitext(output_path)[0] + '_charts.png'
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path


class AdvancedRAGSystem:
    """
    Advanced Retrieval-Augmented Generation system integrating all components for
    intelligent document retrieval, processing, and summarization.
    """
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the advanced RAG system."""
        logger.info(f"Initializing AdvancedRAGSystem with knowledge base path: {knowledge_base_path}")
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_path}")
        
        # Initialize components
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.indexer = SemanticIndexer(use_faiss=faiss_available)
        self.query_processor = QueryProcessor()
        self.summarizer = AbstractiveSummarizer()
        self.conversation_manager = ConversationManager()
        
        # Load and index documents
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}
        self.initialized = False
    
    def initialize(self, force_reload=False):
        """Initialize the system by loading and indexing documents."""
        if self.initialized and not force_reload:
            logger.info("System already initialized")
            return
        
        # Load documents
        logger.info("Loading documents from knowledge base...")
        self.documents = self.document_processor.load_all_documents()
        
        if not self.documents:
            logger.warning("No documents loaded from knowledge base!")
            return
        
        # Create index
        logger.info("Creating semantic index...")
        self.indexer.index_documents(self.documents)
        
        self.initialized = True
        logger.info("System initialization complete")
    
    def add_document(self, file_path):
        """
        Add a new document to the knowledge base and update the index.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document ID if successful, None otherwise
        """
        # Ensure system is initialized
        if not self.initialized:
            self.initialize()
        
        # Load and process the document
        document = self.document_processor.load_document(file_path)
        if not document:
            return None
        
        # Create chunks
        document.create_chunks(self.document_processor.chunker)
        
        # Add to documents dictionary
        self.documents[document.id] = document
        
        # Update index
        self.indexer.index_documents({document.id: document})
        
        return document.id
    
    def process_query(self, query, use_conversation_history=True, conversation_id=None):
        """
        Process a query and generate a response.
        
        Args:
            query: The user's query
            use_conversation_history: Whether to use conversation history for context
            conversation_id: ID of the conversation (if None, use default conversation)
            
        Returns:
            Dict containing query results, response, etc.
        """
        # Ensure system is initialized
        if not self.initialized:
            self.initialize()
        
        if not self.documents:
            return {
                "success": False,
                "error": "No documents in knowledge base",
                "query": query,
                "retrieved_docs": []
            }
        
        # Process the query to understand intent, expand terms, etc.
        query_info = self.query_processor.process_query(query)
        logger.info(f"Processed query. Intent: {query_info['intent']}, expanded: {query_info['expanded_query']}")
        
        # Get conversation context if enabled
        context = None
        if use_conversation_history:
            context = self.conversation_manager.get_conversation_context(query)
        
        # Retrieve relevant chunks using the expanded query
        search_query = query_info["expanded_query"]
        relevant_chunks = self.indexer.hybrid_search(search_query, top_k=10)
        
        if not relevant_chunks:
            return {
                "success": False,
                "error": "No relevant information found",
                "query": query,
                "query_info": query_info,
                "retrieved_docs": []
            }
        
        # Get document IDs for retrieved chunks
        retrieved_docs = []
        seen_docs = set()
        
        for chunk, score in relevant_chunks:
            doc_id = chunk.document.id
            if doc_id not in seen_docs:
                retrieved_docs.append(doc_id)
                seen_docs.add(doc_id)
        
        # Generate summary from retrieved chunks
        summary = self.summarizer.generate_summary(relevant_chunks, query_info)
        
        # Extract source information
        sources = []
        for chunk, _ in relevant_chunks[:5]:  # Limit to top 5 sources
            doc = chunk.document
            source = {
                "id": doc.id,
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_id": chunk.id,
                "relevance": "high" if _ > 0.7 else "medium" if _ > 0.4 else "low"
            }
            sources.append(source)
        
        # Create response
        response = {
            "success": True,
            "query": query,
            "query_info": query_info,
            "response": summary,
            "retrieved_docs": retrieved_docs,
            "sources": sources
        }
        
        # Add to conversation history
        self.conversation_manager.add_interaction(query, summary, retrieved_docs, relevant_chunks)
        
        return response
    
    def generate_pdf_report(self, query, response, retrieved_docs, output_path=None):
        """
        Generate a PDF report containing the query, response, and sources.
        
        Args:
            query: The user's query
            response: The generated response
            retrieved_docs: List of retrieved document IDs
            output_path: Path to save the PDF (if None, use a default location)
            
        Returns:
            Path to the generated PDF
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"rag_report_{timestamp}.pdf"
        
        try:
            # Create a PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='Justify',
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY
            ))
            
            # Title style
            title_style = styles['Heading1']
            title_style.alignment = TA_CENTER
            
            # Create elements for the PDF
            elements = []
            
            # Add title
            elements.append(Paragraph("Question & Answer Report", title_style))
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add query
            elements.append(Paragraph("Query:", styles['Heading2']))
            elements.append(Paragraph(query, styles['Normal']))
            elements.append(Spacer(1, 0.25 * inch))
            
            # Add response
            elements.append(Paragraph("Response:", styles['Heading2']))
            
            # Process response text into paragraphs
            response_paragraphs = response.split("\n\n")
            for para in response_paragraphs:
                elements.append(Paragraph(para, styles['Justify']))
                elements.append(Spacer(1, 0.1 * inch))
            
            elements.append(Spacer(1, 0.25 * inch))
            
            # Add sources
            elements.append(Paragraph("Sources:", styles['Heading2']))
            
            # Create a list of source documents
            for doc_id in retrieved_docs:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    filename = doc.metadata.get("filename", "Unknown")
                    elements.append(Paragraph(f"• {filename}", styles['Normal']))
            
            elements.append(Spacer(1, 0.5 * inch))
            
            # Add footer with timestamp
            footer_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elements.append(Paragraph(footer_text, styles['Normal']))
            
            # Build the PDF
            doc.build(elements)
            
            logger.info(f"PDF report generated at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def save_state(self, path=None):
        """
        Save the system state to a file.
        
        Args:
            path: Path to save the state (if None, use a default location)
            
        Returns:
            Path to the saved state file
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"rag_system_state_{timestamp}.pkl"
        
        try:
            # Create a state dictionary with necessary components
            state = {
                "knowledge_base_path": self.knowledge_base_path,
                "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
                "conversation_history": self.conversation_manager.history,
                "conversation_feedback": self.conversation_manager.feedback,
                "user_preferences": self.conversation_manager.user_preferences,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"System state saved to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
            return None
    
    def load_state(self, path):
        """
        Load the system state from a file.
        
        Args:
            path: Path to the state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load state from file
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Validate state
            if "knowledge_base_path" not in state or "documents" not in state:
                logger.error("Invalid state file: missing required components")
                return False
            
            # Restore knowledge base path
            self.knowledge_base_path = state["knowledge_base_path"]
            
            # Restore documents
            self.documents = {}
            for doc_id, doc_dict in state["documents"].items():
                self.documents[doc_id] = Document.from_dict(doc_dict)
            
            # Restore conversation history and feedback
            if "conversation_history" in state:
                self.conversation_manager.history = state["conversation_history"]
            if "conversation_feedback" in state:
                self.conversation_manager.feedback = state["conversation_feedback"]
            if "user_preferences" in state:
                self.conversation_manager.user_preferences = state["user_preferences"]
            
            # Reindex documents
            self.indexer.index_documents(self.documents)
            
            self.initialized = True
            logger.info(f"System state loaded from: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")
            return False


# Main function to run the RAG system
def main():
    """Main function to run the advanced RAG system."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced RAG System")
    parser.add_argument('--knowledge_base', type=str, default="knowledge_base_docs",
                      help="Path to the knowledge base directory")
    parser.add_argument('--interactive', action='store_true',
                      help="Run in interactive mode")
    parser.add_argument('--query', type=str, 
                      help="Process a single query")
    parser.add_argument('--add_document', type=str,
                      help="Add a document to the knowledge base")
    parser.add_argument('--save_state', type=str,
                      help="Save system state to file")
    parser.add_argument('--load_state', type=str,
                      help="Load system state from file")
    parser.add_argument('--output', type=str,
                      help="Output path for reports")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    print(f"Initializing AdvancedRAGSystem with knowledge base: {args.knowledge_base}")
    rag_system = AdvancedRAGSystem(args.knowledge_base)
    
    # Load state if specified
    if args.load_state:
        success = rag_system.load_state(args.load_state)
        if not success:
            print(f"Failed to load state from: {args.load_state}")
            return
    else:
        rag_system.initialize()
    
    # Add document if specified
    if args.add_document:
        doc_id = rag_system.add_document(args.add_document)
        if doc_id:
            print(f"Document added successfully. ID: {doc_id}")
        else:
            print(f"Failed to add document: {args.add_document}")
    
    # Process a single query if specified
    if args.query:
        result = rag_system.process_query(args.query)
        
        if result["success"]:
            print("\n" + "=" * 80)
            print("QUERY:", args.query)
            print("-" * 80)
            print("RESPONSE:")
            print(result["response"])
            print("-" * 80)
            print("SOURCES:")
            for doc_id in result["retrieved_docs"]:
                if doc_id in rag_system.documents:
                    print(f"- {rag_system.documents[doc_id].metadata.get('filename', doc_id)}")
            print("=" * 80)
            
            # Generate PDF report if output path specified
            if args.output:
                pdf_path = rag_system.generate_pdf_report(
                    args.query, 
                    result["response"], 
                    result["retrieved_docs"], 
                    args.output
                )
                if pdf_path:
                    print(f"\nPDF report generated at: {pdf_path}")
        else:
            print(f"Error: {result['error']}")
    
    # Run in interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit, 'help' for commands.")
        
        while True:
            query = input("\nEnter your query: ")
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  help          - Show this help message")
                print("  exit          - Exit the program")
                print("  add <path>    - Add a document to the knowledge base")
                print("  save <path>   - Save system state to file")
                print("  load <path>   - Load system state from file")
                print("  pdf <path>    - Generate PDF report for the last query")
                continue
            elif query.lower().startswith('add '):
                doc_path = query[4:].strip()
                doc_id = rag_system.add_document(doc_path)
                if doc_id:
                    print(f"Document added successfully. ID: {doc_id}")
                else:
                    print(f"Failed to add document: {doc_path}")
                continue
            elif query.lower().startswith('save '):
                save_path = query[5:].strip()
                saved_path = rag_system.save_state(save_path)
                if saved_path:
                    print(f"System state saved to: {saved_path}")
                else:
                    print("Failed to save system state")
                continue
            elif query.lower().startswith('load '):
                load_path = query[5:].strip()
                success = rag_system.load_state(load_path)
                if success:
                    print(f"System state loaded from: {load_path}")
                else:
                    print(f"Failed to load system state from: {load_path}")
                continue
            elif query.lower().startswith('pdf '):
                pdf_path = query[4:].strip()
                
                # Check if there's a previous query
                if not rag_system.conversation_manager.history:
                    print("No previous query to generate a report for")
                    continue
                
                last_interaction = rag_system.conversation_manager.history[-1]
                pdf_path = rag_system.generate_pdf_report(
                    last_interaction["query"],
                    last_interaction["response"],
                    last_interaction["doc_ids"],
                    pdf_path
                )
                
                if pdf_path:
                    print(f"PDF report generated at: {pdf_path}")
                else:
                    print("Failed to generate PDF report")
                continue
            
            # Process the query
            result = rag_system.process_query(query)
            
            if result["success"]:
                print("\n" + "=" * 80)
                print("RESPONSE:")
                print(result["response"])
                print("-" * 80)
                print("SOURCES:")
                for doc_id in result["retrieved_docs"]:
                    if doc_id in rag_system.documents:
                        print(f"- {rag_system.documents[doc_id].metadata.get('filename', doc_id)}")
                print("=" * 80)
            else:
                print(f"Error: {result['error']}")
    
    # Save state if specified
    if args.save_state:
        saved_path = rag_system.save_state(args.save_state)
        if saved_path:
            print(f"System state saved to: {saved_path}")
        else:
            print(f"Failed to save system state to: {args.save_state}")


# For use in a Jupyter notebook
def create_rag_system(knowledge_base_path="knowledge_base_docs"):
    """Create an advanced RAG system for use in a Jupyter notebook."""
    rag_system = AdvancedRAGSystem(knowledge_base_path)
    rag_system.initialize()
    return rag_system


# Run the system if executed directly
if __name__ == "__main__":
    main()
