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
            content = ''
            metadata = {
                "page_count": len(pdf_reader.pages),
                "file_size_kb": os.path.getsize(file_path) // 1024
            }
            
            # Extract document info if available
            if pdf_reader.metadata:
                try:
                    metadata.update({
                        "title": pdf_reader.metadata.get("/Title", "Unknown"),
                        "author": pdf_reader.metadata.get("/Author", "Unknown"),
                        "creator": pdf_reader.metadata.get("/Creator", "Unknown"),
                        "producer": pdf_reader.metadata.get("/Producer", "Unknown")
                    })
                except:
                    pass
            
            # Extract text content from each page
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text() or ""
                content += page_text + "\n\n"
        
        # Extract tables if enabled
        tables_extracted = False
        if self.table_extraction_enabled:
            try:
                # Use camelot to extract tables
                tables = camelot.read_pdf(file_path, pages='all')
                if len(tables) > 0:
                    tables_extracted = True
                    metadata["table_count"] = len(tables)
                    
                    # Add tables to content
                    content += "\n\n--- TABLES ---\n\n"
                    for i, table in enumerate(tables):
                        # Convert to markdown format for better readability
                        table_df = table.df
                        content += f"Table {i+1}:\n"
                        content += table_df.to_markdown(index=False) if hasattr(table_df, 'to_markdown') else table_df.to_string(index=False)
                        content += "\n\n"
            except Exception as e:
                logger.warning(f"Error extracting tables from PDF: {str(e)}")
        
        # Extract text from images if OCR is enabled and we don't have much text content
        if self.multimodal_enabled and (len(content.strip()) < 1000 or "scanned" in file_path.lower()):
            try:
                # Convert PDF pages to images
                images = pdf2image.convert_from_path(file_path)
                
                # Perform OCR on each image
                ocr_text = []
                for i, img in enumerate(images):
                    # Skip if we already have a lot of text from this page
                    existing_page_text = pdf_reader.pages[i].extract_text() if i < len(pdf_reader.pages) else ""
                    if len(existing_page_text) > 500:  # Skip if we already have substantial text
                        continue
                        
                    # Perform OCR
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        ocr_text.append(f"[OCR Page {i+1}]: {text}")
                
                if ocr_text:
                    content += "\n\n--- OCR EXTRACTED TEXT ---\n\n"
                    content += "\n\n".join(ocr_text)
                    metadata["ocr_applied"] = True
                    metadata["ocr_page_count"] = len(ocr_text)
            except Exception as e:
                logger.warning(f"Error performing OCR on PDF: {str(e)}")
        
        # Extract image descriptions if content is short
        if self.multimodal_enabled and len(content.strip()) < 2000:
            try:
                # Convert PDF pages to images if not already done
                if not 'images' in locals():
                    images = pdf2image.convert_from_path(file_path)
                
                # Add basic image descriptions
                image_descriptions = []
                for i, img in enumerate(images):
                    # Get image properties
                    width, height = img.size
                    # Analyze colors
                    colors = img.getcolors(maxcolors=256)
                    is_grayscale = all(r == g == b for _, (r, g, b) in colors) if colors else False
                    color_mode = "grayscale" if is_grayscale else "color"
                    
                    description = f"[Image {i+1}]: {width}x{height} {color_mode} image"
                    image_descriptions.append(description)
                
                if image_descriptions:
                    content += "\n\n--- IMAGES ---\n\n"
                    content += "\n".join(image_descriptions)
                    metadata["image_count"] = len(image_descriptions)
            except Exception as e:
                logger.warning(f"Error extracting image information from PDF: {str(e)}")
        
        return Document(doc_id=doc_id, file_path=file_path, content=content, metadata=metadata)
        
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        if not text or len(text.strip()) == 0:
            return []
            
        try:
            # Limit text length to avoid memory issues
            text = text[:50000]
            doc = nlp(text)
            
            # Extract noun phrases and entities
            key_terms = []
            
            # Get named entities
            for ent in doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'EVENT'):
                    key_terms.append(ent.text)
            
            # Get noun phrases
            for chunk in doc.noun_chunks:
                # Only include multi-word phrases or important single nouns
                if len(chunk.text.split()) > 1 or (chunk.root.pos_ == 'NOUN' and chunk.root.tag_ not in ('NN', 'NNS')):
                    key_terms.append(chunk.text)
            
            # Extract technical terms using POS patterns
            technical_patterns = [
                [{'POS': 'ADJ'}, {'POS': 'NOUN'}],  # e.g., "neural network"
                [{'POS': 'NOUN'}, {'POS': 'NOUN'}],  # e.g., "database system"
                [{'POS': 'PROPN'}, {'POS': 'PROPN'}]  # e.g., "Google Cloud"
            ]
            
            from spacy.matcher import Matcher
            matcher = Matcher(nlp.vocab)
            for i, pattern in enumerate(technical_patterns):
                matcher.add(f"TECH_{i}", [pattern])
            
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                if span.text.lower() not in STOPWORDS and len(span.text) > 3:
                    key_terms.append(span.text)
            
            # Remove duplicates and sort by length (longer terms first)
            key_terms = list(set(key_terms))
            key_terms.sort(key=lambda x: len(x), reverse=True)
            
            return key_terms[:10]  # Return top 10 terms
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}")
            return []


class VectorStore:
    """
    Manages vector embeddings and similarity search for document chunks.
    Supports multiple embedding methods and efficient retrieval.
    """
    
    def __init__(self, embedding_dim=384):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.documents = {}  # Map of doc_id to Document
        self.chunks = {}  # Map of chunk_id to DocumentChunk
        self.chunk_embeddings = {}  # Map of chunk_id to embedding vector
        
        # Set embedding method based on available packages
        if sentence_transformers_available:
            self.embedding_method = "sentence_transformers"
            self.sentence_model = sentence_model
        else:
            self.embedding_method = "tfidf"
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = None
            self.tfidf_features = None
        
        # Initialize FAISS index if available
        self.use_faiss = faiss_available
        self.index = None
        self.chunk_ids = []  # To maintain order for FAISS
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store and compute embeddings for their chunks.
        
        Args:
            documents: List of Document objects to add
        """
        # Add new documents to the store
        new_chunks = []
        for doc in documents:
            self.documents[doc.id] = doc
            for chunk in doc.chunks:
                self.chunks[chunk.id] = chunk
                new_chunks.append(chunk)
        
        # Compute embeddings for new chunks
        self._compute_embeddings(new_chunks)
        
        # Rebuild index if using FAISS
        if self.use_faiss:
            self._build_faiss_index()
    
    def _compute_embeddings(self, chunks: List[DocumentChunk]):
        """
        Compute embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            return
            
        # Get texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Compute embeddings based on selected method
        if self.embedding_method == "sentence_transformers":
            embeddings = self.sentence_model.encode(texts)
            
            # Store embeddings
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                self.chunk_embeddings[chunk.id] = embeddings[i]
        else:
            # If this is the first batch, fit the vectorizer
            if self.tfidf_matrix is None:
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.tfidf_features = self.vectorizer.get_feature_names_out()
                
                # Store embeddings
                for i, chunk in enumerate(chunks):
                    chunk.embedding = self.tfidf_matrix[i]
                    self.chunk_embeddings[chunk.id] = self.tfidf_matrix[i]
            else:
                # Transform new texts using the existing vectorizer
                new_tfidf = self.vectorizer.transform(texts)
                
                # Store embeddings
                for i, chunk in enumerate(chunks):
                    chunk.embedding = new_tfidf[i]
                    self.chunk_embeddings[chunk.id] = new_tfidf[i]
    
    def _build_faiss_index(self):
        """Build a FAISS index from the chunk embeddings."""
        if not self.use_faiss or not self.chunk_embeddings:
            return
            
        # Get all chunk IDs and embeddings
        self.chunk_ids = list(self.chunk_embeddings.keys())
        if self.embedding_method == "sentence_transformers":
            embeddings = np.array([self.chunk_embeddings[cid] for cid in self.chunk_ids], dtype=np.float32)
        else:
            # Convert sparse matrix to dense for FAISS
            embeddings = np.array([self.chunk_embeddings[cid].toarray()[0] for cid in self.chunk_ids], dtype=np.float32)
        
        # Create and train the index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (cosine on normalized vectors)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
    
    def similarity_search(self, query: str, k: int = 5, filter_func=None) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform a similarity search for the query.
        
        Args:
            query: Query text
            k: Number of top results to return
            filter_func: Optional function to filter chunks (takes a chunk and returns boolean)
            
        Returns:
            List of (DocumentChunk, score) tuples
        """
        if not self.chunk_embeddings:
            return []
        
        # Get query embedding
        if self.embedding_method == "sentence_transformers":
            query_embedding = self.sentence_model.encode(query)
            
            # Perform search
            if self.use_faiss and self.index is not None:
                # Normalize query vector
                query_embedding_np = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(query_embedding_np)
                
                # Search in FAISS index
                scores, indices = self.index.search(query_embedding_np, k*2)  # Get more results to allow for filtering
                
                # Get chunks from results
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunk_ids):
                        chunk_id = self.chunk_ids[idx]
                        chunk = self.chunks[chunk_id]
                        
                        # Apply filter if provided
                        if filter_func is None or filter_func(chunk):
                            results.append((chunk, float(score)))
                
                # Limit to k results after filtering
                return results[:k]
            else:
                # Compute similarities manually
                results = []
                for chunk_id, embedding in self.chunk_embeddings.items():
                    chunk = self.chunks[chunk_id]
                    
                    # Apply filter if provided
                    if filter_func is not None and not filter_func(chunk):
                        continue
                    
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                    results.append((chunk, float(similarity)))
                
                # Sort by similarity (descending) and return top k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
        else:
            # TFIDF-based search
            query_embedding = self.vectorizer.transform([query])[0]
            
            results = []
            for chunk_id, embedding in self.chunk_embeddings.items():
                chunk = self.chunks[chunk_id]
                
                # Apply filter if provided
                if filter_func is not None and not filter_func(chunk):
                    continue
                
                # Compute cosine similarity
                similarity = cosine_similarity(query_embedding, embedding)[0][0]
                results.append((chunk, float(similarity)))
            
            # Sort by similarity (descending) and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
    
    def save(self, file_path: str):
        """Save the vector store to disk."""
        # Create data structure to save
        data = {
            "embedding_method": self.embedding_method,
            "embedding_dim": self.embedding_dim,
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "chunk_ids": self.chunk_ids
        }
        
        # Save data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
        # Save embeddings separately if using TFIDF (sparse matrices)
        if self.embedding_method == "tfidf":
            with open(f"{file_path}.tfidf", 'wb') as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "tfidf_features": self.tfidf_features
                }, f)
    
    @classmethod
    def load(cls, file_path: str):
        """Load a vector store from disk."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create new vector store
        vector_store = cls(embedding_dim=data["embedding_dim"])
        vector_store.embedding_method = data["embedding_method"]
        vector_store.chunk_ids = data["chunk_ids"]
        
        # Reconstruct documents and chunks
        documents = {}
        chunks = {}
        for doc_id, doc_dict in data["documents"].items():
            doc = Document.from_dict(doc_dict)
            documents[doc_id] = doc
            for chunk in doc.chunks:
                chunks[chunk.id] = chunk
        
        vector_store.documents = documents
        vector_store.chunks = chunks
        
        # Load embeddings for TFIDF method
        if vector_store.embedding_method == "tfidf":
            with open(f"{file_path}.tfidf", 'rb') as f:
                tfidf_data = pickle.load(f)
            
            vector_store.vectorizer = tfidf_data["vectorizer"]
            vector_store.tfidf_matrix = tfidf_data["tfidf_matrix"]
            vector_store.tfidf_features = tfidf_data["tfidf_features"]
            
            # Reconstruct chunk embeddings
            for i, chunk_id in enumerate(vector_store.chunk_ids):
                if i < vector_store.tfidf_matrix.shape[0]:
                    vector_store.chunk_embeddings[chunk_id] = vector_store.tfidf_matrix[i]
        
class HybridSearcher:
    """
    Implements hybrid search combining BM25 and semantic search for more effective retrieval.
    This combines the strengths of lexical and semantic search methods.
    """
    
    def __init__(self, vector_store=None):
        """
        Initialize the hybrid searcher.
        
        Args:
            vector_store: Optional VectorStore instance for semantic search
        """
        self.vector_store = vector_store
        self.bm25 = None
        self.corpus = []  # Text of each chunk
        self.chunk_map = {}  # Maps corpus index to chunk ID
        self.tokenized_corpus = []
        self.initialized = False
    
    def initialize(self, chunks: List[DocumentChunk]):
        """
        Initialize the BM25 index with document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        self.corpus = [chunk.text for chunk in chunks]
        self.chunk_map = {i: chunk.id for i, chunk in enumerate(chunks)}
        
        # Tokenize corpus for BM25
        self.tokenized_corpus = [text.lower().split() for text in self.corpus]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.initialized = True
    
    def search(self, query: str, k: int = 5, semantic_weight: float = 0.5, filter_func=None) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform hybrid search combining BM25 and semantic search.
        
        Args:
            query: Query text
            k: Number of results to return
            semantic_weight: Weight to give semantic search results (0-1)
            filter_func: Optional function to filter chunks
            
        Returns:
            List of (DocumentChunk, score) tuples
        """
        if not self.initialized or not self.bm25:
            logger.warning("HybridSearcher not initialized. Call initialize() first.")
            return []
        
        lexical_weight = 1.0 - semantic_weight
        
        # Normalize weights
        total_weight = lexical_weight + semantic_weight
        if total_weight > 0:
            lexical_weight = lexical_weight / total_weight
            semantic_weight = semantic_weight / total_weight
        else:
            lexical_weight = 0.5
            semantic_weight = 0.5
        
        # Get results from BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1 range
        max_bm25_score = max(bm25_scores) if bm25_scores.size > 0 else 1.0
        if max_bm25_score > 0:
            bm25_scores = bm25_scores / max_bm25_score
        
        # Get results from semantic search if available
        semantic_scores = {}
        if self.vector_store and semantic_weight > 0:
            semantic_results = self.vector_store.similarity_search(query, k=k*2, filter_func=filter_func)
            for chunk, score in semantic_results:
                semantic_scores[chunk.id] = score
        
        # Combine results
        combined_scores = {}
        
        # Add BM25 scores
        for i, score in enumerate(bm25_scores):
            chunk_id = self.chunk_map[i]
            chunk = self.vector_store.chunks[chunk_id] if self.vector_store else None
            
            # Skip if filtered out
            if filter_func and chunk and not filter_func(chunk):
                continue
                
            combined_scores[chunk_id] = score * lexical_weight
        
        # Add semantic scores
        for chunk_id, score in semantic_scores.items():
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += score * semantic_weight
            else:
                combined_scores[chunk_id] = score * semantic_weight
        
        # Sort by combined score
        sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)
        
        # Construct result list
        results = []
        for chunk_id in sorted_chunk_ids[:k]:
            chunk = self.vector_store.chunks[chunk_id] if self.vector_store else None
            if chunk:
                results.append((chunk, combined_scores[chunk_id]))
        
        return results


class QueryProcessor:
    """
    Handles query analysis, refinement, and expansion for more effective retrieval.
    """
    
    def __init__(self):
        """Initialize the query processor."""
        self.synonyms_enabled = True
        self.query_expansion_enabled = True
        self.history = []  # List of past queries and their refined versions
    
    def process_query(self, query: str, query_context: Dict = None) -> Dict[str, Any]:
        """
        Analyze and refine a user query.
        
        Args:
            query: The original user query
            query_context: Optional context information (e.g., conversation history)
            
        Returns:
            Dictionary with refined query and query analysis
        """
        # Basic preprocessing
        clean_query = self._preprocess_query(query)
        
        # Analyze query
        analysis = self._analyze_query(clean_query)
        
        # Refine query based on analysis
        refined_query = self._refine_query(clean_query, analysis, query_context)
        
        # Expand query with synonyms if enabled
        expanded_query = self._expand_query(refined_query) if self.query_expansion_enabled else refined_query
        
        # Add to history
        self.history.append({
            "original": query,
            "cleaned": clean_query,
            "refined": refined_query,
            "expanded": expanded_query,
            "analysis": analysis
        })
        
        return {
            "original_query": query,
            "refined_query": refined_query,
            "expanded_query": expanded_query,
            "analysis": analysis
        }
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove special characters that might interfere with search
        query = re.sub(r'[^\w\s?!.,-]', '', query)
        
        return query
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to determine its type, structure, and key terms.
        
        Returns:
            Dictionary containing query analysis
        """
        analysis = {
            "query_type": None,
            "key_terms": [],
            "entities": [],
            "question_type": None,
            "focus": None
        }
        
        # Parse with spaCy
        doc = nlp(query)
        
        # Extract key terms
        key_terms = []
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN', 'VERB') and not token.is_stop:
                key_terms.append(token.lemma_)
        
        analysis["key_terms"] = key_terms
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
        
        analysis["entities"] = entities
        
        # Determine query type (question, command, keyword)
        if query.endswith('?') or query.lower().startswith(('what', 'who', 'where', 'when', 'why', 'how')):
            analysis["query_type"] = "question"
            
            # Determine question type
            first_word = doc[0].text.lower() if len(doc) > 0 else ""
            question_types = {
                'what': 'definition',
                'who': 'person',
                'where': 'location',
                'when': 'time',
                'why': 'reason',
                'how': 'process'
            }
            
            if first_word in question_types:
                analysis["question_type"] = question_types[first_word]
        
        elif doc[0].pos_ == 'VERB' if len(doc) > 0 else False:
            analysis["query_type"] = "command"
        else:
            analysis["query_type"] = "keyword"
        
        # Determine query focus (main subject)
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj', 'pobj') and token.pos_ in ('NOUN', 'PROPN'):
                analysis["focus"] = token.text
                break
        
        return analysis
    
    def _refine_query(self, query: str, analysis: Dict[str, Any], query_context: Dict = None) -> str:
        """
        Refine the query based on analysis and context.
        
        Args:
            query: The preprocessed query
            analysis: The query analysis dictionary
            query_context: Optional context information
            
        Returns:
            Refined query string
        """
        refined_query = query
        
        # Don't refine if it's just a simple keyword query
        if analysis["query_type"] == "keyword" and len(query.split()) <= 3:
            return query
        
        # Add focus for certain question types that might be implicit
        if analysis["query_type"] == "question" and analysis["question_type"] == "definition" and not analysis["focus"]:
            # Look for noun after "what is" pattern
            match = re.search(r'what\s+is\s+(?:a|an|the)?\s*([a-z0-9_\- ]+)', query.lower())
            if match:
                focus_term = match.group(1).strip()
                if focus_term and not focus_term in refined_query:
                    refined_query = f"{refined_query} about {focus_term}"
        
        # Incorporate context if available
        if query_context and 'conversation_history' in query_context:
            history = query_context['conversation_history']
            
            # If query is very short and looks like a follow-up
            if len(query.split()) <= 5 and not analysis["focus"] and history:
                last_query = history[-1].get('query', '')
                last_query_analysis = self._analyze_query(last_query)
                
                # Check if this might be a follow-up question
                follow_up_indicators = ['they', 'them', 'it', 'this', 'that', 'these', 'those', 'their', 'its']
                if any(word in query.lower().split() for word in follow_up_indicators):
                    # Add topic from previous query for context
                    if last_query_analysis.get("focus"):
                        refined_query = f"{refined_query} about {last_query_analysis['focus']}"
        
        return refined_query
    
    def _expand_query(self, query: str) -> str:
        """
        Expand the query with synonyms and related terms.
        
        Args:
            query: The refined query
            
        Returns:
            Expanded query string
        """
        # Parse query
        doc = nlp(query)
        
        # Extract content words
        content_words = [token.text for token in doc 
                       if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'PROPN') 
                       and not token.is_stop
                       and len(token.text) > 3]
        
        # Limit to 3 most important terms to avoid query drift
        if len(content_words) > 3:
            # Prioritize nouns and named entities
            nouns_and_entities = [token.text for token in doc 
                                if token.pos_ in ('NOUN', 'PROPN') 
                                and not token.is_stop]
            if len(nouns_and_entities) >= 3:
                content_words = nouns_and_entities[:3]
            else:
                content_words = content_words[:3]
        
        # Get synonyms for each word
        expanded_terms = set()
        for word in content_words:
            synonyms = self._get_synonyms(word)
            # Add top 2 synonyms at most
            expanded_terms.update(synonyms[:2])
        
        # Remove any terms already in the query
        query_words = query.lower().split()
        expanded_terms = [term for term in expanded_terms 
                        if term.lower() not in query_words 
                        and all(term.lower() not in qw for qw in query_words)]
        
        # Add expanded terms to query
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            return expanded_query
            
        return query
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: The word to find synonyms for
            
        Returns:
            List of synonym strings
        """
        synonyms = set()
        
        # Get synonyms from WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) > 3:
                    synonyms.add(synonym)
        
        # Limit to 5 synonyms
        return list(synonyms)[:5]


class ImprovedSummaryGenerator:
    """
    Generates high-quality, coherent summaries using advanced NLP techniques.
    Supports both extractive and abstractive summarization methods.
    """
    
    def __init__(self):
        """Initialize the summary generator."""
        self.language = "english"
        self.stop_words = STOPWORDS
        
        # Initialize extractive summarization components
        if sumy_available:
            self.stemmer = Stemmer(self.language)
            self.sumy_stop_words = get_stop_words(self.language)
            
            # Initialize extractive summarizers
            self.lexrank = LexRankSummarizer(self.stemmer)
            self.lexrank.stop_words = self.sumy_stop_words
            
            self.lsa = LsaSummarizer(self.stemmer)
            self.lsa.stop_words = self.sumy_stop_words
        
        # Check if transformers-based summarization is available
        self.abstractive_enabled = transformers_summarization_available
    
    def generate_summary(self, texts: List[str], query: str, max_length: int = 500, use_abstractive: bool = None) -> str:
        """
        Generate a comprehensive, coherent summary from multiple texts based on the query.
        
        Args:
            texts: List of document texts to summarize
            query: The user's query
            max_length: Target maximum length of summary in words
            use_abstractive: Whether to use abstractive summarization (if available)
                             If None, will automatically decide based on content
            
        Returns:
            A coherent summary of the key information relevant to the query
        """
        if not texts:
            return "No relevant information found."
            
        try:
            # Process and clean the texts
            processed_texts = self._preprocess_texts(texts)
            if not processed_texts:
                return "No relevant information found after processing."
                
            # Extract query intent and key concepts
            query_concepts = self._extract_key_concepts(query)
            
            # Determine if we should use abstractive summarization
            if use_abstractive is None:
                # Use abstractive for shorter, more focused content
                total_length = sum(len(text.split()) for text in processed_texts)
                use_abstractive = self.abstractive_enabled and total_length < 3000
            
            # Generate summary using appropriate method
            if use_abstractive and self.abstractive_enabled:
                summary = self._generate_abstractive_summary(processed_texts, query, query_concepts, max_length)
            else:
                summary = self._generate_extractive_summary(processed_texts, query, query_concepts, max_length)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fall back to basic extractive summarization
            return self._generate_basic_summary(texts, query)
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Clean and normalize the input texts."""
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                continue
                
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove excessive punctuation
            text = re.sub(r'([.!?])\1+', r'\1', text)
            
            # Remove content in brackets if it looks like citations or references
            text = re.sub(r'\[\d+\]', '', text)
            
            processed_texts.append(text.strip())
            
        return processed_texts
    
    def _extract_key_concepts(self, query: str) -> Dict[str, Any]:
        """Extract key concepts and intent from the query."""
        doc = nlp(query)
        
        concepts = {
            'keywords': [],
            'entities': [],
            'root_verb': None,
            'question_type': None,
            'original_query': query
        }
        
        # Extract keywords (important nouns and adjectives)
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop:
                concepts['keywords'].append(token.lemma_)
            elif token.pos_ == 'ADJ' and token.is_alpha and len(token.text) > 2:
                concepts['keywords'].append(token.lemma_)
        
        # Extract named entities
        for ent in doc.ents:
            concepts['entities'].append((ent.text, ent.label_))
            
        # Determine the main verb (often indicates the action requested)
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                concepts['root_verb'] = token.lemma_
                
        # Determine question type if it's a question
        question_words = {'what': 'definition', 'how': 'process', 'why': 'reason', 
                         'when': 'time', 'where': 'location', 'who': 'person'}
        
        if len(doc) > 0:
            first_word = doc[0].text.lower()
            if first_word in question_words:
                concepts['question_type'] = question_words[first_word]
            elif '?' in query:
                concepts['question_type'] = 'general'
            
        return concepts
    
    def _generate_extractive_summary(self, texts: List[str], query: str, query_concepts: Dict[str, Any], max_length: int) -> str:
        """
        Generate a summary using extractive methods, which select and organize the most relevant sentences.
        
        Args:
            texts: List of preprocessed document texts
            query: The user's query
            query_concepts: Dictionary with query analysis
            max_length: Target maximum length in words
            
        Returns:
            Extractive summary text
        """
        # Combine texts into a single document
        combined_text = "\n\n".join(texts)
        
        # Extract information units (sentences)
        sentences = nltk.sent_tokenize(combined_text)
        
        # Score sentences by relevance to query
        scored_sentences = self._score_sentences_by_relevance(sentences, query, query_concepts)
        
        # Remove redundancies
        unique_sentences = self._remove_redundant_sentences(scored_sentences)
        
        # Select top sentences within length limit
        selected_sentences = self._select_within_length(unique_sentences, max_length)
        
        # Organize selected sentences
        organized_sentences = self._organize_sentences(selected_sentences, query_concepts)
        
        # Generate the final summary text
        summary = self._format_summary(organized_sentences, query, query_concepts)
        
        return summary
    
    def _score_sentences_by_relevance(self, sentences: List[str], query: str, query_concepts: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Score sentences by their relevance to the query."""
        scored_sentences = []
        
        # Extract query terms for matching
        query_terms = set(query_concepts['keywords'])
        entity_terms = set(entity for entity, _ in query_concepts['entities'])
        all_query_terms = query_terms.union(entity_terms)
        
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            # Create a spaCy doc for the sentence
            sentence_doc = nlp(sentence)
            
            # Initialize score components
            term_match_score = 0.0
            semantic_score = 0.0
            question_type_score = 0.0
            position_score = 0.0
            
            # Term matching score
            sentence_terms = set()
            for token in sentence_doc:
                if not token.is_stop and token.is_alpha and len(token.text) > 2:
                    sentence_terms.add(token.lemma_)
            
            # Calculate term overlap
            matching_terms = sentence_terms.intersection(all_query_terms)
            if all_query_terms:
                term_match_score = len(matching_terms) / len(all_query_terms)
            
            # Adjust score based on question type
            if query_concepts['question_type']:
                # Boost sentences that match specific question types
                if query_concepts['question_type'] == 'definition':
                    pattern = r'\b(is|are|refers to|defined as|means)\b'
                    if re.search(pattern, sentence.lower()):
                        question_type_score = 0.3
                        
                elif query_concepts['question_type'] == 'process':
                    process_words = ['step', 'process', 'procedure', 'method', 'approach']
                    if any(word in sentence.lower() for word in process_words):
                        question_type_score = 0.2
                        
                elif query_concepts['question_type'] == 'reason':
                    reason_words = ['because', 'since', 'therefore', 'due to', 'result']
                    if any(word in sentence.lower() for word in reason_words):
                        question_type_score = 0.2
            
            # Calculate final score
            final_score = 0.5 * term_match_score + 0.3 * semantic_score + 0.2 * question_type_score
            
            # Only include sentences with some relevance
            if final_score > 0.1:
                scored_sentences.append((sentence, final_score))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        return scored_sentences
    
    def _remove_redundant_sentences(self, scored_sentences: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Remove redundant sentences to avoid repetition."""
        if not scored_sentences:
            return []
            
        # Sort sentences by score (already done, but just to be sure)
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Extract just the sentences for processing
        sentences = [sent for sent, _ in sorted_sentences]
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            doc_i = nlp(sentences[i])
            
            for j in range(i+1, len(sentences)):
                doc_j = nlp(sentences[j])
                
                # Calculate similarity
                try:
                    # Use spaCy's similarity if vectors are available
                    if has_vectors:
                        similarity = doc_i.similarity(doc_j)
                    else:
                        # Fallback to word overlap
                        words_i = set(token.text.lower() for token in doc_i 
                                    if token.is_alpha and not token.is_stop)
                        words_j = set(token.text.lower() for token in doc_j 
                                    if token.is_alpha and not token.is_stop)
                        
                        if words_i and words_j:
                            similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                        else:
                            similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                except:
                    # If similarity calculation fails, assume low similarity
                    similarity_matrix[i, j] = 0.1
                    similarity_matrix[j, i] = 0.1
        
        # Initialize list to keep track of which sentences to include
        include_mask = np.ones(len(sentences), dtype=bool)
        
        # Iterate through sentences in order of score
        for i in range(len(sentences)):
            if not include_mask[i]:
                continue  # Skip already excluded sentences
                
            # Check similarity with all higher-scoring sentences
            for j in range(i+1, len(sentences)):
                if not include_mask[j]:
                    continue  # Skip already excluded sentences
                    
                # If very similar, exclude the lower-scoring sentence
                if similarity_matrix[i, j] > 0.6:  # Threshold for similarity
                    include_mask[j] = False
        
        # Create filtered list of non-redundant sentences with their scores
        unique_sentences = [(sentences[i], sorted_sentences[i][1]) 
                          for i in range(len(sentences)) 
                          if include_mask[i]]
        
        return unique_sentences
    
    def _select_within_length(self, scored_sentences: List[Tuple[str, float]], max_length: int) -> List[Tuple[str, float]]:
        """Select top sentences that fit within the target length."""
        if not scored_sentences:
            return []
            
        current_length = 0
        selected_sentences = []
        
        for sentence, score in scored_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                selected_sentences.append((sentence, score))
                current_length += sentence_length
            
            if current_length >= max_length:
                break
                
        return selected_sentences
    
    def _organize_sentences(self, scored_sentences: List[Tuple[str, float]], query_concepts: Dict[str, Any]) -> List[str]:
        """
        Organize selected sentences into a coherent structure based on query type.
        
        Returns:
            List of organized sentences
        """
        if not scored_sentences:
            return []
            
        # Extract just the sentences (discard scores)
        sentences = [sent for sent, _ in scored_sentences]
        
        # For definition queries, organize by definition, elaboration, examples
        if query_concepts['question_type'] == 'definition':
            definitions = []
            elaborations = []
            examples = []
            
            for sentence in sentences:
                lowercase = sentence.lower()
                
                # Check for definition patterns
                if any(pattern in lowercase for pattern in ['is a', 'refers to', 'defined as', 'means']):
                    definitions.append(sentence)
                # Check for examples
                elif any(word in lowercase for word in ['example', 'instance', 'such as', 'like']):
                    examples.append(sentence)
                else:
                    elaborations.append(sentence)
            
            # Combine in logical order
            organized = definitions + elaborations + examples
            return organized
            
        # For process queries, try to identify sequential steps
        elif query_concepts['question_type'] == 'process':
            # Look for sentences with sequential markers
            step_sentences = []
            other_sentences = []
            
            for sentence in sentences:
                if re.search(r'\b(first|second|third|next|then|finally|step|phase)\b', sentence.lower()):
                    step_sentences.append(sentence)
                else:
                    other_sentences.append(sentence)
            
            # Put step sentences first, then other sentences
            return step_sentences + other_sentences
            
        # For reason queries, organize by claims and reasoning
        elif query_concepts['question_type'] == 'reason':
            claims = []
            reasons = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['because', 'since', 'due to', 'reason']):
                    reasons.append(sentence)
                else:
                    claims.append(sentence)
            
            # Put claims first, then reasons
            return claims + reasons
            
        # Default organization is by relevance (already sorted by score)
        return sentences
    
    def _format_summary(self, sentences: List[str], query: str, query_concepts: Dict[str, Any]) -> str:
        """Format the final summary with introduction and conclusion."""
        if not sentences:
            return "No relevant information was found to answer this query."
        
        # Create introduction
        introduction = self._create_introduction(query, query_concepts)
        
        # Create paragraphs from sentences
        paragraphs = self._create_paragraphs(sentences)
        
        # Create conclusion if appropriate
        conclusion = self._create_conclusion(query, query_concepts)
        
        # Combine all parts
        summary_parts = [introduction]
        summary_parts.extend(paragraphs)
        if conclusion:
            summary_parts.append(conclusion)
        
        return "\n\n".join(summary_parts)
    
    def _create_paragraphs(self, sentences: List[str]) -> List[str]:
        """Group sentences into coherent paragraphs."""
        if not sentences:
            return []
            
        # If few sentences, return as a single paragraph
        if len(sentences) <= 3:
            return [" ".join(sentences)]
            
        # Initialize paragraphs
        paragraphs = []
        current_paragraph = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            previous_sentence = sentences[i-1]
            
            # Check if sentences are related
            # This is a simple heuristic - could be improved with semantic analysis
            sentence_1_doc = nlp(previous_sentence)
            sentence_2_doc = nlp(current_sentence)
            
            # Get content words from each sentence
            words_1 = set(token.text.lower() for token in sentence_1_doc 
                        if token.is_alpha and not token.is_stop)
            words_2 = set(token.text.lower() for token in sentence_2_doc 
                        if token.is_alpha and not token.is_stop)
            
            # Calculate overlap
            if words_1 and words_2:
                overlap = len(words_1.intersection(words_2)) / len(words_1.union(words_2))
            else:
                overlap = 0
            
            # If sentences are related and paragraph not too long, add to current paragraph
            if overlap > 0.2 and len(current_paragraph) < 5:
                current_paragraph.append(current_sentence)
            else:
                # Start a new paragraph
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = [current_sentence]
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        return paragraphs
    
    def _create_introduction(self, query: str, query_concepts: Dict[str, Any]) -> str:
        """Create an introductory sentence based on the query."""
        # Extract query focus (main subject)
        focus = None
        for entity, _ in query_concepts['entities']:
            focus = entity
            break
            
        if not focus and query_concepts['keywords']:
            focus = query_concepts['keywords'][0]
        
        # Generate introduction
        if focus:
            if query_concepts['question_type'] == 'definition':
                return f"Here is information about what {focus} is:"
            elif query_concepts['question_type'] == 'process':
                return f"Here is information about how {focus} works:"
            elif query_concepts['question_type'] == 'reason':
                return f"Here is information about why {focus} happens:"
            else:
                return f"Here is information about {focus}:"
        else:
            return "Here is the relevant information based on your query:"
    
    def _create_conclusion(self, query: str, query_concepts: Dict[str, Any]) -> str:
        """Create a concluding sentence if appropriate."""
        # Only add conclusion for certain query types
        if query_concepts['question_type'] in ['reason', 'process']:
            return "This summary represents the key points addressing your query based on the available information."
        
        return ""  # No conclusion for other query types
    
    def _generate_abstractive_summary(self, texts: List[str], query: str, query_concepts: Dict[str, Any], max_length: int) -> str:
        """
        Generate a summary using abstractive (generative) methods, which create new text instead of extracting.
        
        Args:
            texts: List of preprocessed document texts
            query: The user's query
            query_concepts: Dictionary with query analysis
            max_length: Target maximum length in words
            
        Returns:
            Abstractive summary text
        """
        if not self.abstractive_enabled:
            return self._generate_extractive_summary(texts, query, query_concepts, max_length)
        
        try:
            # Combine texts with appropriate weighting based on query relevance
            combined_text = self._prepare_text_for_abstractive(texts, query, query_concepts)
            
            # Ensure the combined text isn't too long for the model
            # BART has a 1024 token limit for input
            max_tokens = 1000
            tokens = summarization_tokenizer(combined_text, return_tensors="pt")
            if tokens.input_ids.shape[1] > max_tokens:
                # If too long, use extractive summarization to reduce length first
                # Create a shorter extractive summary to use as input
                extractive_summary = self._generate_extractive_summary(texts, query, query_concepts, max_length=max_tokens//4)
                combined_text = extractive_summary
            
            # Prepare prompt with query focus
            introduction = ""
            if query_concepts['question_type'] == 'definition':
                if query_concepts['keywords']:
                    introduction = f"Information about what {query_concepts['keywords'][0]} is: "
            elif query_concepts['question_type'] == 'process':
                if query_concepts['keywords']:
                    introduction = f"Information about how {query_concepts['keywords'][0]} works: "
            
            # Append introduction if we have one
            if introduction:
                combined_text = introduction + combined_text
            
            # Generate summary
            # Calculate max_new_tokens based on max_length
            # Rule of thumb: 1 word ≈ 1.3 tokens for English
            max_new_tokens = min(int(max_length * 1.3), 500)  # Cap at 500 tokens
            
            # Generate summary
            summary_output = summarizer(
                combined_text,
                max_length=max_new_tokens,
                min_length=int(max_new_tokens/3),  # At least 1/3 of max length
                do_sample=False,  # Deterministic generation
                truncation=True
            )
            
            # Extract summary text
            summary_text = summary_output[0]['summary_text']
            
            # Post-process summary
            summary_text = self._post_process_abstractive_summary(summary_text, query, query_concepts)
            
            return summary_text
            
        except Exception as e:
            logger.error(f"Error generating abstractive summary: {str(e)}")
            # Fall back to extractive summarization
            return self._generate_extractive_summary(texts, query, query_concepts, max_length)
    
    def _prepare_text_for_abstractive(self, texts: List[str], query: str, query_concepts: Dict[str, Any]) -> str:
        """Prepare text for abstractive summarization by filtering and organizing content."""
        # Combine texts
        combined_text = "\n\n".join(texts)
        
        # Extract key sentences
        sentences = nltk.sent_tokenize(combined_text)
        
        # Get query terms for filtering
        query_terms = set(query_concepts['keywords'])
        entity_terms = set(entity for entity, _ in query_concepts['entities'])
        all_query_terms = query_terms.union(entity_terms)
        
        # Filter to sentences with query relevance
        if all_query_terms:
            relevant_sentences = []
            for sentence in sentences:
                # Convert to lowercase for case-insensitive matching
                lower_sentence = sentence.lower()
                
                # Check if any query term appears in the sentence
                if any(term.lower() in lower_sentence for term in all_query_terms):
                    relevant_sentences.append(sentence)
                    
            # If we have enough relevant sentences, use only those
            if len(relevant_sentences) >= 3:
                combined_text = " ".join(relevant_sentences)
        
        return combined_text
    
    def _post_process_abstractive_summary(self, summary: str, query: str, query_concepts: Dict[str, Any]) -> str:
        """Clean up and improve the generated abstractive summary."""
        # Clean up any tokenization artifacts
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure first letter is capitalized
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure the summary ends with proper punctuation
        if summary and summary[-1] not in ['.', '!', '?']:
            summary += '.'
        
        # If summary is very short, combine with extractive approach
        if len(summary.split()) < 25:
            extractive_summary = self._generate_basic_summary(texts=[summary], query=query)
            summary = summary + "\n\n" + extractive_summary
        
        return summary
    
    def _generate_basic_summary(self, texts: List[str], query: str) -> str:
        """
        Generate a basic summary using traditional extractive methods.
        Used as a fallback when other methods fail.
        """
        if not texts:
            return "No relevant information found."
            
        if not sumy_available:
            # Ultra simple fallback
            sentences = []
            for text in texts:
                sentences.extend(nltk.sent_tokenize(text))
            
            # Take first 5 sentences as summary
            if sentences:
                return " ".join(sentences[:5])
            else:
                return "No relevant information found."
        
        try:
            # Combine texts
            combined_text = "\n\n".join(texts)
            
            # Parse the text
            parser = PlaintextParser.from_string(combined_text, Tokenizer(self.language))
            
            # Generate summary
            summary_sentences = [str(s) for s in self.lexrank(parser.document, 5)]
            
            if not summary_sentences:
                return "No relevant information could be extracted."
                
            return " ".join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error generating basic summary: {str(e)}")
            
            # Ultra simple fallback
            sentences = []
            for text in texts:
                sentences.extend(nltk.sent_tokenize(text)[:3])  # Take first 3 sentences from each text
            
            if sentences:
                return " ".join(sentences[:5])  # Take at most 5 sentences total
            else:
                return "No relevant information found."


class ConversationContext:
    """
    Manages conversation history and context for multi-turn interactions.
    Helps with follow-up questions and maintaining coherence across the conversation.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize the conversation context.
        
        Args:
            max_history: Maximum number of turns to keep in history
        """
        self.history = []
        self.max_history = max_history
        self.entities = {}  # Named entities mentioned in conversation
        self.topics = {}  # Topics discussed with frequency counts
    
    def add_exchange(self, query: str, response: str, query_analysis: Dict = None, retrieved_docs: List[str] = None):
        """
        Add a query-response exchange to the conversation history.
        
        Args:
            query: The user's query
            response: The system's response
            query_analysis: Optional analysis of the query
            retrieved_docs: Optional list of retrieved document IDs
        """
        exchange = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "analysis": query_analysis,
            "retrieved_docs": retrieved_docs
        }
        
        # Update entities and topics from query analysis
        if query_analysis:
            # Add entities
            if 'entities' in query_analysis:
                for entity, entity_type in query_analysis['entities']:
                    if entity in self.entities:
                        self.entities[entity]['count'] += 1
                    else:
                        self.entities[entity] = {
                            'type': entity_type,
                            'count': 1,
                            'first_seen': len(self.history)
                        }
            
            # Add topics
            if 'keywords' in query_analysis:
                for keyword in query_analysis['keywords']:
                    if keyword in self.topics:
                        self.topics[keyword] += 1
                    else:
                        self.topics[keyword] = 1
        
        # Add to history
        self.history.append(exchange)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Get relevant context for the current query.
        
        Args:
            query: The current user query
            
        Returns:
            Dictionary with context information
        """
        # Check if this might be a follow-up question
        is_followup = self._is_followup(query)
        
        # Get references to entities and topics from history
        referenced_entities = self._get_referenced_entities(query)
        active_topics = self._get_active_topics()
        
        # Get recent exchanges
        recent_exchanges = self.history[-3:] if self.history else []
        
        # Create context dictionary
        context = {
            "is_followup": is_followup,
            "referenced_entities": referenced_entities,
            "active_topics": active_topics,
            "recent_exchanges": recent_exchanges,
            "conversation_length": len(self.history)
        }
        
        return context
    
    def _is_followup(self, query: str) -> bool:
        """Determine if a query is likely a follow-up question."""
        if not self.history:
            return False
            
        # Check for pronouns that might refer to previous context
        followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'their']
        query_words = query.lower().split()
        
        # If query is very short and contains followup indicators
        if len(query_words) <= 5 and any(word in followup_indicators for word in query_words):
            return True
        
        # If query starts with a verb or doesn't contain a named entity, might be followup
        doc = nlp(query)
        if len(doc) > 0 and doc[0].pos_ == 'VERB':
            return True
        
        # If no entities in query but entities in history
        query_has_entities = any(ent.label_ in ('PERSON', 'ORG', 'GPE', 'PRODUCT') for ent in doc.ents)
        if not query_has_entities and self.entities:
            return True
        
        return False
    
    def _get_referenced_entities(self, query: str) -> List[Dict[str, Any]]:
        """Find entities from history that might be referenced in the query."""
        if not self.entities:
            return []
            
        query_doc = nlp(query)
        query_tokens = [token.text.lower() for token in query_doc]
        
        # Check for pronouns that might refer to entities
        has_pronouns = any(token.lower() in ['it', 'this', 'that', 'they', 'them', 'their'] 
                          for token in query_tokens)
        
        # Get entities that might be referenced
        referenced = []
        
        # If query has pronouns, consider recently mentioned entities
        if has_pronouns and self.history:
            # Get entities from most recent exchange
            last_exchange = self.history[-1]
            if 'analysis' in last_exchange and 'entities' in last_exchange['analysis']:
                for entity, entity_type in last_exchange['analysis']['entities']:
                    if entity in self.entities:
                        referenced.append({
                            'entity': entity,
                            'type': entity_type,
                            'relevance': 'high'
                        })
        
        # Otherwise, check for partial matches with known entities
        else:
            for entity, info in self.entities.items():
                # Check if entity words appear in the query
                entity_words = entity.lower().split()
                if any(word in query.lower() for word in entity_words if len(word) > 3):
                    referenced.append({
                        'entity': entity,
                        'type': info['type'],
                        'relevance': 'medium',
                        'count': info['count']
                    })
        
        return referenced
    
    def _get_active_topics(self) -> List[str]:
        """Get currently active topics from conversation history."""
        if not self.topics:
            return []
            
        # Get topics sorted by frequency and recency
        sorted_topics = sorted(self.topics.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 topics
        return [topic for topic, count in sorted_topics[:5]]
    
    def clear(self):
        """Clear conversation history and context."""
        self.history = []
        self.entities = {}
        self.topics = {}


class EvaluationMetrics:
    """
    Provides methods for evaluating the performance of the RAG system.
    Supports both automatic and user feedback-based metrics.
    """
    
    def __init__(self):
        """Initialize the evaluation metrics."""
        self.query_metrics = {}  # Metrics for individual queries
        self.overall_metrics = {
            "queries_processed": 0,
            "successful_queries": 0,
            "retrieval_precision": [],
            "retrieval_recall": [],
            "summary_quality": [],
            "user_ratings": [],
            "response_time": []
        }
    
    def record_query_metrics(self, query_id: str, metrics: Dict[str, Any]):
        """
        Record metrics for a specific query.
        
        Args:
            query_id: Unique identifier for the query
            metrics: Dictionary of metrics to record
        """
        self.query_metrics[query_id] = metrics
        
        # Update overall metrics
        self.overall_metrics["queries_processed"] += 1
        
        if metrics.get("success", False):
            self.overall_metrics["successful_queries"] += 1
            
        if "retrieval_precision" in metrics:
            self.overall_metrics["retrieval_precision"].append(metrics["retrieval_precision"])
            
        if "retrieval_recall" in metrics:
            self.overall_metrics["retrieval_recall"].append(metrics["retrieval_recall"])
            
        if "summary_quality" in metrics:
            self.overall_metrics["summary_quality"].append(metrics["summary_quality"])
            
        if "user_rating" in metrics:
            self.overall_metrics["user_ratings"].append(metrics["user_rating"])
            
        if "response_time" in metrics:
            self.overall_metrics["response_time"].append(metrics["response_time"])
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for a generated summary.
        
        Args:
            prediction: Generated summary
            reference: Reference (gold standard) summary
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(prediction, reference)[0]
            
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except:
            # If ROUGE calculation fails, return zeros
            return {
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0
            }
    
    def get_retrieval_metrics(self, retrieved_docs: List[str], relevant_docs: List[str]) -> Dict[str, float]:
        """
        Calculate retrieval metrics.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs (ground truth)
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not relevant_docs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # Calculate precision
        if retrieved_docs:
            precision = len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs)
        else:
            precision = 0.0
            
        # Calculate recall
        recall = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def record_user_feedback(self, query_id: str, rating: int, feedback: str = None):
        """
        Record user feedback for a query.
        
        Args:
            query_id: Unique identifier for the query
            rating: User rating (1-5)
            feedback: Optional textual feedback
        """
        if query_id in self.query_metrics:
            self.query_metrics[query_id]["user_rating"] = rating
            if feedback:
                self.query_metrics[query_id]["user_feedback"] = feedback
                
            # Update overall metrics
            self.overall_metrics["user_ratings"].append(rating)
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Get overall system metrics.
        
        Returns:
            Dictionary with aggregated metrics
        """
        metrics = {
            "queries_processed": self.overall_metrics["queries_processed"],
            "success_rate": (self.overall_metrics["successful_queries"] / 
                           max(1, self.overall_metrics["queries_processed"]))
        }
        
        # Calculate average metrics
        if self.overall_metrics["retrieval_precision"]:
            metrics["avg_retrieval_precision"] = sum(self.overall_metrics["retrieval_precision"]) / len(self.overall_metrics["retrieval_precision"])
        
        if self.overall_metrics["retrieval_recall"]:
            metrics["avg_retrieval_recall"] = sum(self.overall_metrics["retrieval_recall"]) / len(self.overall_metrics["retrieval_recall"])
        
        if self.overall_metrics["summary_quality"]:
            metrics["avg_summary_quality"] = sum(self.overall_metrics["summary_quality"]) / len(self.overall_metrics["summary_quality"])
        
        if self.overall_metrics["user_ratings"]:
            metrics["avg_user_rating"] = sum(self.overall_metrics["user_ratings"]) / len(self.overall_metrics["user_ratings"])
        
        if self.overall_metrics["response_time"]:
            metrics["avg_response_time"] = sum(self.overall_metrics["response_time"]) / len(self.overall_metrics["response_time"])
        
        return metrics
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a human-readable evaluation report.
        
        Returns:
            Formatted report string
        """
        overall_metrics = self.get_overall_metrics()
        
        report = "# RAG System Evaluation Report\n\n"
        
        # Overall statistics
        report += "## Overall Performance\n\n"
        report += f"- Total queries processed: {overall_metrics['queries_processed']}\n"
        report += f"- Success rate: {overall_metrics['success_rate']:.2f}\n"
        
        if "avg_retrieval_precision" in overall_metrics:
            report += f"- Average retrieval precision: {overall_metrics['avg_retrieval_precision']:.2f}\n"
        
        if "avg_retrieval_recall" in overall_metrics:
            report += f"- Average retrieval recall: {overall_metrics['avg_retrieval_recall']:.2f}\n"
        
        if "avg_summary_quality" in overall_metrics:
            report += f"- Average summary quality: {overall_metrics['avg_summary_quality']:.2f}\n"
        
        if "avg_user_rating" in overall_metrics:
            report += f"- Average user rating: {overall_metrics['avg_user_rating']:.2f}\n"
        
        if "avg_response_time" in overall_metrics:
            report += f"- Average response time: {overall_metrics['avg_response_time']:.2f} seconds\n"
        
        # Recent queries analysis
        report += "\n## Recent Queries\n\n"
        
        recent_queries = list(self.query_metrics.items())[-5:]
        for query_id, metrics in recent_queries:
            report += f"### Query: {metrics.get('query', 'Unknown')}\n"
            report += f"- Success: {metrics.get('success', False)}\n"
            report += f"- Response time: {metrics.get('response_time', 0):.2f} seconds\n"
            
            if "retrieval_precision" in metrics:
                report += f"- Retrieval precision: {metrics['retrieval_precision']:.2f}\n"
            
            if "user_rating" in metrics:
                report += f"- User rating: {metrics['user_rating']}\n"
            
            if "user_feedback" in metrics:
                report += f"- User feedback: {metrics['user_feedback']}\n"
            
            report += "\n"
        
        report += "\n"
        return report


class AdvancedRAGSystem:
    """
    Main RAG system that integrates all components for document processing,
    retrieval, summarization, and conversation management.
    """
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the advanced RAG system.
        
        Args:
            knowledge_base_path: Path to the knowledge base directory
        """
        self.knowledge_base_path = knowledge_base_path
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_path}")
        
        # Initialize components
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.vector_store = VectorStore()
        self.hybrid_searcher = HybridSearcher(self.vector_store)
        self.query_processor = QueryProcessor()
        self.summary_generator = ImprovedSummaryGenerator()
        self.conversation_context = ConversationContext()
        self.evaluation_metrics = EvaluationMetrics()
        
        # System state
        self.initialized = False
        self.has_documents = False
    
    def initialize(self):
        """Initialize the system by loading documents and building indexes."""
        # Load documents
        logger.info("Initializing the RAG system...")
        documents = self.document_processor.load_all_documents()
        
        if documents:
            # Add documents to vector store
            self.vector_store.add_documents(list(documents.values()))
            
            # Initialize hybrid searcher
            all_chunks = []
            for doc in documents.values():
                all_chunks.extend(doc.chunks)
            
            self.hybrid_searcher.initialize(all_chunks)
            
            self.has_documents = True
            self.initialized = True
            logger.info(f"System initialized with {len(documents)} documents and {len(all_chunks)} chunks")
        else:
            logger.warning("No documents found in knowledge base. System initialized with empty state.")
            self.initialized = True
    
    def process_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's query
            conversation_id: Optional conversation identifier for context
            
        Returns:
            Dictionary with query results
        """
        # Ensure system is initialized
        if not self.initialized:
            self.initialize()
            
        # Check if we have documents
        if not self.has_documents:
            return {
                "success": False,
                "error": "No documents in knowledge base. Please add documents before querying.",
                "query": query
            }
        
        # Start timing for performance metrics
        start_time = time.time()
        
        # Generate a query ID
        query_id = f"q_{str(uuid.uuid4())[:8]}"
        
        try:
            # Get conversation context if available
            context = self.conversation_context.get_context_for_query(query)
            
            # Process and analyze query
            processed_query = self.query_processor.process_query(query, context)
            
            # Perform hybrid search
            search_results = self.hybrid_searcher.search(
                processed_query["expanded_query"],
                k=7,  # Retrieve more results for better coverage
                semantic_weight=0.7  # Weight semantic search higher
            )
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return {
                    "success": False,
                    "error": "No relevant information found in the knowledge base.",
                    "query": query,
                    "query_id": query_id,
                    "processed_query": processed_query
                }
            
            # Extract relevant texts from search results
            retrieved_chunks = [chunk for chunk, _ in search_results]
            retrieved_docs = list(set(chunk.document.id for chunk in retrieved_chunks))
            retrieved_texts = [chunk.text for chunk in retrieved_chunks]
            
            # Generate summary
            summary = self.summary_generator.generate_summary(
                retrieved_texts,
                processed_query["original_query"],
                max_length=500,  # Target summary length
                use_abstractive=None  # Auto-decide based on content
            )
            
            # Extract key terms
            all_content = " ".join(retrieved_texts)
            key_terms = self.document_processor.extract_key_terms(all_content)
            
            # Format response
            response = self._format_response(
                summary, 
                key_terms, 
                processed_query["original_query"],
                processed_query["analysis"]
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add to conversation context
            self.conversation_context.add_exchange(
                query=query,
                response=response,
                query_analysis=processed_query["analysis"],
                retrieved_docs=retrieved_docs
            )
            
            # Record metrics
            self.evaluation_metrics.record_query_metrics(
                query_id=query_id,
                metrics={
                    "query": query,
                    "success": True,
                    "response_time": response_time,
                    "retrieval_count": len(retrieved_chunks),
                    "doc_count": len(retrieved_docs)
                }
            )
            
            # Build result dictionary
            result = {
                "success": True,
                "query": query,
                "query_id": query_id,
                "response": response,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_docs": retrieved_docs,
                "processed_query": processed_query,
                "execution_time": response_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            
            # Record failed query
            self.evaluation_metrics.record_query_metrics(
                query_id=query_id,
                metrics={
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
            )
            
            return {
                "success": False,
                "error": f"An error occurred while processing your query: {str(e)}",
                "query": query,
                "query_id": query_id
            }
    
    def _format_response(self, summary: str, key_terms: List[str], query: str, query_analysis: Dict[str, Any]) -> str:
        """Format the final response with summary and key terms."""
        response = f"# Summary: {query}\n\n"
        response += summary
        
        if key_terms:
            response += "\n\n## Key Terms\n"
            for term in key_terms:
                # Capitalize the first letter of each term
                formatted_term = term[0].upper() + term[1:] if term else ""
                response += f"- {formatted_term}\n"
        
        return response
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a new document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with status information
        """
        # Ensure system is initialized
        if not self.initialized:
            self.initialize()
            
        try:
            # Process document
            document = self.document_processor.load_document(file_path)
            
            if not document:
                return {
                    "success": False,
                    "error": f"Failed to load document: {file_path}",
                    "file_path": file_path
                }
            
            # Add to vector store
            self.vector_store.add_documents([document])
            
            # Update hybrid searcher
            self.hybrid_searcher.initialize(document.chunks)
            
            self.has_documents = True
            
            return {
                "success": True,
                "message": f"Document added successfully: {os.path.basename(file_path)}",
                "document_id": document.id,
                "chunk_count": len(document.chunks)
            }
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error adding document: {str(e)}",
                "file_path": file_path
            }
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """
        Generate a PDF report of the response.
        
        Args:
            query: User query
            response: System response
            retrieved_docs: List of retrieved document IDs
            
        Returns:
            Path to the generated PDF file
        """
        # Create a temporary directory for the PDF
        logger.info("Generating PDF report")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(temp_dir, f"response_{timestamp}.pdf")
                
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
                    fontSize=11,
                    leading=14,
                    alignment=TA_JUSTIFY
                ))
                
                # Add title style
                title_style = ParagraphStyle(
                    name='Title',
                    fontName='Helvetica-Bold',
                    fontSize=16,
                    leading=20,
                    alignment=TA_CENTER,
                    spaceAfter=20
                )
                
                # Create elements for the PDF
                elements = []
                
                # Add title
                elements.append(Paragraph(f"Query: {query}", title_style))
                elements.append(Spacer(1, 0.25 * inch))
                
                # Process markdown in response
                paragraphs = response.split("\n\n")
                
                for paragraph in paragraphs:
                    if paragraph.startswith("# "):
                        # Main heading
                        heading_text = paragraph[2:].strip()
                        elements.append(Paragraph(heading_text, styles['Heading1']))
                        elements.append(Spacer(1, 0.15 * inch))
                    elif paragraph.startswith("## "):
                        # Subheading
                        subheading_text = paragraph[3:].strip()
                        elements.append(Paragraph(subheading_text, styles['Heading2']))
                        elements.append(Spacer(1, 0.1 * inch))
                    elif paragraph.startswith("- "):
                        # List item
                        list_text = paragraph[2:].strip()
                        elements.append(Paragraph("• " + list_text, styles['Normal']))
                        elements.append(Spacer(1, 0.05 * inch))
                    else:
                        # Regular paragraph
                        elements.append(Paragraph(paragraph, styles['Justify']))
                        elements.append(Spacer(1, 0.1 * inch))
                
                # Add sources
                doc_names = []
                for doc_id in retrieved_docs:
                    if doc_id in self.document_processor.documents:
                        doc = self.document_processor.documents[doc_id]
                        doc_names.append(doc.metadata.get("filename", doc_id))
                
                if doc_names:
                    elements.append(Paragraph("Sources", styles['Heading2']))
                    elements.append(Spacer(1, 0.1 * inch))
                    
                    for name in doc_names:
                        elements.append(Paragraph("• " + name, styles['Normal']))
                        elements.append(Spacer(1, 0.05 * inch))
                
                # Add footer with timestamp
                footer_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                elements.append(Spacer(1, 0.5 * inch))
                elements.append(Paragraph(footer_text, styles['Normal']))
                
                # Build the PDF
                try:
                    doc.build(elements)
                except AttributeError as e:
                    logger.error(f"Error building PDF: {str(e)}")
                    if "'str' object has no attribute 'build'" in str(e):
                        logger.error("This likely indicates an issue with reportlab's SimpleDocTemplate initialization")
                        # Try an alternative approach
                        doc = SimpleDocTemplate(output_path)
                        doc.build(elements)
                
                # In Vertex AI Workbench notebook environment
                try:
                    final_path = f"/home/jupyter/response_{timestamp}.pdf"
                    os.system(f"cp {output_path} {final_path}")
                    return final_path
                except Exception as ex:
                    logger.error(f"Error copying PDF to final location: {str(ex)}")
                    return output_path
                
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return f"Error generating PDF: {str(e)}"
    
    def save_state(self, directory: str = None):
        """
        Save the system state to disk.
        
        Args:
            directory: Optional directory to save state. If None, uses knowledge_base_path.
        """
        if directory is None:
            directory = os.path.join(self.knowledge_base_path, "system_state")
            
        os.makedirs(directory, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(os.path.join(directory, "vector_store.pkl"))
        
        # Save system metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "document_count": len(self.document_processor.documents),
            "chunk_count": len(self.vector_store.chunks),
            "has_documents": self.has_documents,
            "initialized": self.initialized
        }
        
        with open(os.path.join(directory, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"System state saved to {directory}")
        
        return directory
    
    @classmethod
    def load_state(cls, directory: str, knowledge_base_path: str = None):
        """
        Load a system from saved state.
        
        Args:
            directory: Directory with saved state
            knowledge_base_path: Optional knowledge base path. If None, uses path from metadata.
            
        Returns:
            AdvancedRAGSystem instance
        """
        # Load metadata
        try:
            with open(os.path.join(directory, "metadata.json"), 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logger.error(f"Metadata file not found in {directory}")
            raise ValueError(f"Invalid state directory: {directory}")
            
        # Create new system
        if knowledge_base_path is None:
            knowledge_base_path = os.path.dirname(directory)
            
        system = cls(knowledge_base_path)
        
        # Load vector store
        vector_store_path = os.path.join(directory, "vector_store.pkl")
        if os.path.exists(vector_store_path):
            try:
                system.vector_store = VectorStore.load(vector_store_path)
                
                # Reconstruct document processor's documents from vector store
                system.document_processor.documents = system.vector_store.documents
                
                # Update hybrid searcher
                all_chunks = []
                for doc in system.document_processor.documents.values():
                    all_chunks.extend(doc.chunks)
                
                system.hybrid_searcher = HybridSearcher(system.vector_store)
                system.hybrid_searcher.initialize(all_chunks)
                
                system.has_documents = bool(system.document_processor.documents)
                system.initialized = True
                
                logger.info(f"System state loaded from {directory}")
                
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                raise
        else:
            logger.warning(f"Vector store file not found in {directory}")
            system.initialize()
        
        return system


# Main function to run the RAG system
def main():
    """Main function to run the RAG system."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing Advanced RAG system with knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
    # Initialize the RAG system
    rag_system = AdvancedRAGSystem(KNOWLEDGE_BASE_PATH)
    rag_system.initialize()
    
    # Example usage
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            break
        
        print("\nProcessing your query...\n")
        
        # Process the query
        result = rag_system.process_query(query)
        
        if result["success"]:
            print("=" * 80)
            print("RESPONSE:")
            print("-" * 80)
            print(result["response"])
            print("=" * 80)
            print(f"\nSources: {', '.join([chunk.document.metadata.get('filename', chunk.document.id) for chunk in result['retrieved_chunks'][:3]])}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            
            # Generate PDF report
            try:
                pdf_path = rag_system.generate_pdf_report(
                    query,
                    result["response"],
                    result["retrieved_docs"]
                )
                print(f"\nPDF report generated at: {pdf_path}")
            except Exception as e:
                print(f"\nError generating PDF report: {str(e)}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save system state before exiting
    try:
        state_dir = rag_system.save_state()
        print(f"\nSystem state saved to: {state_dir}")
    except Exception as e:
        print(f"\nError saving system state: {str(e)}")


# For use in a Jupyter notebook
def create_rag_system(knowledge_base_path="knowledge_base_docs"):
    """Create and return a RAG system instance for use in a notebook."""
    system = AdvancedRAGSystem(knowledge_base_path)
    system.initialize()
    return system


def process_query_and_generate_pdf(rag_system, query):
    """Process a query and generate a PDF report."""
    result = rag_system.process_query(query)
    
    if result["success"]:
        print("=" * 80)
        print("RESPONSE:")
        print("-" * 80)
        print(result["response"])
        print("=" * 80)
        print(f"\nSources: {', '.join([chunk.document.metadata.get('filename', chunk.document.id) for chunk in result['retrieved_chunks'][:3]])}")
        
        # Generate PDF report
        try:
            pdf_path = rag_system.generate_pdf_report(
                query,
                result["response"],
                result["retrieved_docs"]
            )
            print(f"\nPDF report generated at: {pdf_path}")
            return result["response"], pdf_path
        except Exception as e:
            print(f"\nError generating PDF report: {str(e)}")
            return result["response"], None
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return None, None


# Run the system if executed directly
if __name__ == "__main__":
    main()
