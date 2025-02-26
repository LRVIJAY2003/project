# project
    
"""
Enhanced Retrieval-Augmented Generation (RAG) System
---------------------------------------------------
This improved system processes documents from a knowledge base, retrieves relevant information based on user queries,
and generates concise, coherent summaries using advanced NLP techniques.

Key Enhancements:
- Hybrid retrieval combining semantic and lexical search (BM25)
- Adaptive document chunking for better context handling
- Query expansion and reformulation for improved results
- Conversational context support for multi-turn interactions
- Comprehensive evaluation framework
- Enhanced summarization with redundancy elimination
- Improved document processing with metadata extraction
"""

# ===== First, install required packages =====
import sys
import subprocess
import importlib.util

def install_packages():
    """Install or upgrade required packages."""
    packages = [
        'nltk',
        'spacy',
        'scikit-learn',
        'numpy',
        'pandas',
        'python-docx',
        'PyPDF2',
        'reportlab',
        'sumy',
        'rank_bm25',  # Added for BM25 ranking
        'rouge',      # Added for evaluation
    ]
    
    # Optional but recommended package for better summarization
    optional_packages = [
        'sentence-transformers'
    ]
    
    print("Checking and installing required packages...")
    
    for package in packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  → Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
            print(f"  ✓ {package} installed successfully")
    
    print("\nChecking optional packages (recommended for better performance)...")
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  → Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"  ✓ {package} installed successfully")
            except:
                print(f"  ✗ Could not install {package}. Will use fallback methods.")

# Install required packages
install_packages()

# Now import the necessary modules
import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from datetime import datetime
import string
import textwrap
import random

# Import BM25 for lexical search
from rank_bm25 import BM25Okapi

# Download required NLTK data
print("\nDownloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords, wordnet
STOPWORDS = set(stopwords.words('english'))

# Download required spaCy models
print("\nChecking spaCy models...")
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

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
    print("  ✓ Sentence Transformers available for enhanced semantic understanding")
    # Initialize the model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    sentence_transformers_available = False
    print("  ✗ Sentence Transformers not available. Will use TF-IDF for semantic similarity.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_System")

# Import sumy for traditional summarization (as fallback)
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    sumy_available = True
except ImportError:
    sumy_available = False
    print("  ✗ Sumy summarization library not available. Will use basic summarization.")

# Import rouge for evaluation
try:
    from rouge import Rouge
    rouge_available = True
    print("  ✓ Rouge metrics available for evaluation")
except ImportError:
    rouge_available = False
    print("  ✗ Rouge metrics not available. Will use basic evaluation methods.")


class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    def __init__(self, 
                 text: str, 
                 doc_id: str, 
                 chunk_id: int,
                 start_pos: int = 0,
                 is_title: bool = False,
                 is_heading: bool = False,
                 metadata: Dict[str, Any] = None):
        """Initialize a document chunk."""
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.start_pos = start_pos
        self.is_title = is_title
        self.is_heading = is_heading
        self.metadata = metadata or {}
        # Will be set after embedding creation
        self.embedding = None
        
    def __repr__(self):
        """String representation of the chunk."""
        return f"DocumentChunk(doc_id={self.doc_id}, chunk_id={self.chunk_id}, len={len(self.text)}, {'title' if self.is_title else 'heading' if self.is_heading else 'text'})"
    
    def to_dict(self):
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "start_pos": self.start_pos,
            "is_title": self.is_title,
            "is_heading": self.is_heading,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create chunk from dictionary."""
        return cls(
            text=data["text"],
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            start_pos=data["start_pos"],
            is_title=data["is_title"],
            is_heading=data["is_heading"],
            metadata=data["metadata"]
        )


class DocumentProcessor:
    """Handles document processing, indexing, and retrieval with enhanced chunking."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the document processor."""
        logger.info(f"Initializing DocumentProcessor with path: {knowledge_base_path}")
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.document_chunks = {}  # Will store chunked documents
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1  # Set to 1 to handle cases with few documents
        )
        self.doc_embeddings = {}
        self.chunk_embeddings = {}
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_chunk_map = []
    
    def load_all_documents(self) -> Dict[str, str]:
        """Load all documents from the knowledge base."""
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path, exist_ok=True)
            logger.warning(f"Created knowledge base directory: {self.knowledge_base_path}")
            
        all_files = glob.glob(os.path.join(self.knowledge_base_path, '*.*'))
        
        if not all_files:
            logger.warning(f"No files found in knowledge base path: {self.knowledge_base_path}")
            return self.documents
            
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.txt':
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        self.documents[file_name] = f.read()
                
                elif file_extension == '.docx':
                    doc = docx.Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    self.documents[file_name] = content
                
                elif file_extension == '.pdf':
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ''
                        for page_num in range(len(pdf_reader.pages)):
                            content += pdf_reader.pages[page_num].extract_text()
                        self.documents[file_name] = content
                
                else:
                    logger.warning(f"Unsupported file type: {file_extension} for file {file_name}")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
        return self.documents
    
    def process_all_documents(self):
        """Process all documents including chunking and indexing."""
        if not self.documents:
            self.load_all_documents()
            
        if not self.documents:
            logger.warning("No documents to process")
            return
            
        logger.info("Processing all documents into chunks...")
        
        # Create chunks for all documents
        self.document_chunks = {}
        for doc_id, content in self.documents.items():
            chunks = self.create_semantic_chunks(content, doc_id)
            self.document_chunks[doc_id] = chunks
            
        # Create embeddings for all chunks
        self.create_chunk_embeddings()
        
        # Create BM25 index for lexical search
        self.create_bm25_index()
        
        logger.info(f"Processed {len(self.documents)} documents into {sum(len(chunks) for chunks in self.document_chunks.values())} chunks")
    
    def create_semantic_chunks(self, text: str, doc_id: str, 
                               min_chunk_size: int = 100, 
                               max_chunk_size: int = 500) -> List[DocumentChunk]:
        """Chunk text into semantically coherent pieces."""
        if not text:
            return []
            
        # Extract potential title
        lines = text.split('\n')
        title = None
        if lines and len(lines[0].strip()) < 200 and not lines[0].strip().endswith('.'):
            title = lines[0].strip()
            text = '\n'.join(lines[1:])
            
        # First, extract headings and their content
        heading_pattern = re.compile(r'^(#+|\d+\.|\*+|\d+\)|\w+\))\s+(.+)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(text))
        
        chunks = []
        chunk_id = 0
        
        # Add title as a special chunk if present
        if title:
            chunks.append(DocumentChunk(
                text=title,
                doc_id=doc_id,
                chunk_id=chunk_id,
                is_title=True,
                metadata={"position": "title"}
            ))
            chunk_id += 1
        
        # If no headings, split by paragraphs and sentences
        if not headings:
            # Split into paragraphs
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para.split())
                
                # If paragraph is too large, split it into sentences
                if para_size > max_chunk_size:
                    sentences = nltk.sent_tokenize(para)
                    for sent in sentences:
                        sent_size = len(sent.split())
                        
                        if current_size + sent_size <= max_chunk_size:
                            current_chunk.append(sent)
                            current_size += sent_size
                        else:
                            if current_chunk:
                                combined_text = ' '.join(current_chunk)
                                chunks.append(DocumentChunk(
                                    text=combined_text,
                                    doc_id=doc_id,
                                    chunk_id=chunk_id
                                ))
                                chunk_id += 1
                            current_chunk = [sent]
                            current_size = sent_size
                
                # Otherwise, try to add the paragraph to the current chunk
                elif current_size + para_size <= max_chunk_size:
                    current_chunk.append(para)
                    current_size += para_size
                else:
                    # Current chunk is full, start a new one
                    if current_chunk:
                        combined_text = ' '.join(current_chunk)
                        chunks.append(DocumentChunk(
                            text=combined_text,
                            doc_id=doc_id,
                            chunk_id=chunk_id
                        ))
                        chunk_id += 1
                    current_chunk = [para]
                    current_size = para_size
            
            # Don't forget the last chunk
            if current_chunk:
                combined_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    text=combined_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
        
        # If headings exist, use them to guide chunking
        else:
            last_end = 0
            
            # Process each heading and its content
            for i, match in enumerate(headings):
                start = match.start()
                
                # If there's text before the first heading, process it
                if i == 0 and start > 0:
                    pretext = text[:start].strip()
                    if pretext:
                        pretext_chunks = self._chunk_text_by_size(
                            pretext, doc_id, chunk_id, min_chunk_size, max_chunk_size
                        )
                        chunks.extend(pretext_chunks)
                        chunk_id += len(pretext_chunks)
                
                # Extract the heading text
                heading_text = match.group(2).strip()
                
                # Create a chunk for the heading itself
                chunks.append(DocumentChunk(
                    text=heading_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    start_pos=start,
                    is_heading=True,
                    metadata={"level": len(match.group(1).strip())}
                ))
                chunk_id += 1
                
                # Determine where this section ends (at the next heading or end of text)
                end = headings[i+1].start() if i < len(headings) - 1 else len(text)
                
                # Extract section content (exclude the heading itself)
                section_content = text[match.end():end].strip()
                
                if section_content:
                    # Chunk the section content
                    section_chunks = self._chunk_text_by_size(
                        section_content, doc_id, chunk_id, min_chunk_size, max_chunk_size
                    )
                    chunks.extend(section_chunks)
                    chunk_id += len(section_chunks)
                
                last_end = end
            
            # Process any text after the last heading
            if last_end < len(text):
                remaining_text = text[last_end:].strip()
                if remaining_text:
                    remaining_chunks = self._chunk_text_by_size(
                        remaining_text, doc_id, chunk_id, min_chunk_size, max_chunk_size
                    )
                    chunks.extend(remaining_chunks)
                    chunk_id += len(remaining_chunks)
        
        return chunks
    
    def _chunk_text_by_size(self, text: str, doc_id: str, start_chunk_id: int, 
                            min_chunk_size: int, max_chunk_size: int) -> List[DocumentChunk]:
        """Helper function to chunk text based on size constraints."""
        chunks = []
        chunk_id = start_chunk_id
        
        # Split into paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            # If paragraph is too large, split it into sentences
            if para_size > max_chunk_size:
                sentences = nltk.sent_tokenize(para)
                for sent in sentences:
                    sent_size = len(sent.split())
                    
                    if current_size + sent_size <= max_chunk_size:
                        current_chunk.append(sent)
                        current_size += sent_size
                    else:
                        if current_chunk:
                            combined_text = ' '.join(current_chunk)
                            chunks.append(DocumentChunk(
                                text=combined_text,
                                doc_id=doc_id,
                                chunk_id=chunk_id
                            ))
                            chunk_id += 1
                        current_chunk = [sent]
                        current_size = sent_size
            
            # Otherwise, try to add the paragraph to the current chunk
            elif current_size + para_size <= max_chunk_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                # Current chunk is full, start a new one
                if current_chunk:
                    combined_text = ' '.join(current_chunk)
                    chunks.append(DocumentChunk(
                        text=combined_text,
                        doc_id=doc_id,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                current_chunk = [para]
                current_size = para_size
        
        # Don't forget the last chunk
        if current_chunk:
            combined_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                text=combined_text,
                doc_id=doc_id,
                chunk_id=chunk_id
            ))
            chunk_id += 1
            
        return chunks
    
    def create_chunk_embeddings(self):
        """Create embeddings for all document chunks."""
        if not self.document_chunks:
            logger.warning("No document chunks to embed")
            return
            
        logger.info("Creating embeddings for document chunks...")
        
        all_chunks = []
        chunk_texts = []
        
        # Collect all chunks and their texts
        for doc_id, chunks in self.document_chunks.items():
            all_chunks.extend(chunks)
            chunk_texts.extend([chunk.text for chunk in chunks])
        
        # Create embeddings based on available models
        if sentence_transformers_available:
            # Use sentence transformers for better semantic embeddings
            embeddings = sentence_model.encode(chunk_texts)
            
            # Store embeddings with their chunks
            for i, chunk in enumerate(all_chunks):
                chunk.embedding = embeddings[i]
                
        else:
            # Fallback to TF-IDF
            try:
                tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
                
                # Store embeddings with their chunks
                for i, chunk in enumerate(all_chunks):
                    chunk.embedding = tfidf_matrix[i]
                    
            except Exception as e:
                logger.error(f"Error creating chunk embeddings: {str(e)}")
                
        logger.info(f"Created embeddings for {len(all_chunks)} chunks")
    
    def create_bm25_index(self):
        """Create BM25 index for lexical search."""
        if not self.document_chunks:
            logger.warning("No document chunks for BM25 indexing")
            return
            
        logger.info("Creating BM25 index for document chunks...")
        
        self.bm25_corpus = []
        self.bm25_chunk_map = []
        
        # Collect all chunks and tokenize for BM25
        for doc_id, chunks in self.document_chunks.items():
            for chunk in chunks:
                # Tokenize text for BM25
                tokenized_text = nltk.word_tokenize(chunk.text.lower())
                # Remove stopwords and punctuation
                tokenized_text = [token for token in tokenized_text 
                                if token not in STOPWORDS and token not in string.punctuation]
                
                self.bm25_corpus.append(tokenized_text)
                self.bm25_chunk_map.append(chunk)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"Created BM25 index with {len(self.bm25_corpus)} documents")
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Combine semantic and lexical search for better retrieval."""
        if not self.document_chunks:
            self.process_all_documents()
            
        if not self.document_chunks:
            logger.warning("No document chunks available for search")
            return []
            
        # Process query
        processed_query = self._preprocess_query(query)
        
        # Get semantic search results
        semantic_results = self._semantic_search(processed_query, top_k * 2)
        
        # Get BM25 lexical search results
        lexical_results = self._lexical_search(processed_query, top_k * 2)
        
        # Combine results with score normalization and reranking
        combined_results = self._combine_search_results(semantic_results, lexical_results, processed_query)
        
        # Return top_k results
        return combined_results[:top_k]
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query for search."""
        # Basic cleaning
        query = query.strip()
        
        # Remove excessive punctuation
        query = re.sub(r'([.!?])\1+', r'\1', query)
        
        return query
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Perform semantic search using embeddings."""
        if sentence_transformers_available:
            # Create query embedding using sentence transformers
            query_embedding = sentence_model.encode(query)
            
            # Calculate similarity with all chunks
            results = []
            
            for doc_id, chunks in self.document_chunks.items():
                for chunk in chunks:
                    if chunk.embedding is not None:
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, chunk.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                        )
                        results.append((chunk, float(similarity)))
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        else:
            # Fallback to TF-IDF
            try:
                query_vector = self.vectorizer.transform([query])
                
                results = []
                
                for doc_id, chunks in self.document_chunks.items():
                    for chunk in chunks:
                        if chunk.embedding is not None:
                            similarity = cosine_similarity(query_vector, chunk.embedding)[0][0]
                            results.append((chunk, float(similarity)))
                
                # Sort by similarity (highest first)
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
                
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                return []
    
    def _lexical_search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Perform lexical search using BM25."""
        if self.bm25_index is None:
            logger.warning("BM25 index not created yet")
            return []
            
        # Tokenize query
        query_tokens = nltk.word_tokenize(query.lower())
        query_tokens = [token for token in query_tokens 
                      if token not in STOPWORDS and token not in string.punctuation]
        
        # Get BM25 scores
        try:
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with chunks and scores
            results = []
            for i, score in enumerate(scores):
                results.append((self.bm25_chunk_map[i], score))
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in lexical search: {str(e)}")
            return []
    
    def _combine_search_results(self, 
                                semantic_results: List[Tuple[DocumentChunk, float]], 
                                lexical_results: List[Tuple[DocumentChunk, float]],
                                query: str) -> List[Tuple[DocumentChunk, float]]:
        """Combine and rerank semantic and lexical search results."""
        # Normalize scores to [0, 1] range
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            min_semantic = min(score for _, score in semantic_results)
            score_range = max_semantic - min_semantic
            if score_range > 0:
                semantic_results = [
                    (chunk, (score - min_semantic) / score_range) 
                    for chunk, score in semantic_results
                ]
        
        if lexical_results:
            max_lexical = max(score for _, score in lexical_results)
            min_lexical = min(score for _, score in lexical_results)
            score_range = max_lexical - min_lexical
            if score_range > 0:
                lexical_results = [
                    (chunk, (score - min_lexical) / score_range) 
                    for chunk, score in lexical_results
                ]
        
        # Combine results with weights
        combined = {}
        
        # Default weights
        semantic_weight = 0.7
        lexical_weight = 0.3
        
        # Adjust weights based on query characteristics
        if '?' in query:  # Questions often benefit from semantic search
            semantic_weight = 0.8
            lexical_weight = 0.2
        elif any(term in query.lower() for term in ['how', 'why', 'explain']):
            semantic_weight = 0.8
            lexical_weight = 0.2
        elif len(query.split()) <= 3:  # Short queries often benefit from lexical search
            semantic_weight = 0.5
            lexical_weight = 0.5
            
        # Add semantic results
        for chunk, score in semantic_results:
            chunk_id = (chunk.doc_id, chunk.chunk_id)
            combined[chunk_id] = {"chunk": chunk, "score": score * semantic_weight}
            
        # Add lexical results
        for chunk, score in lexical_results:
            chunk_id = (chunk.doc_id, chunk.chunk_id)
            if chunk_id in combined:
                combined[chunk_id]["score"] += score * lexical_weight
            else:
                combined[chunk_id] = {"chunk": chunk, "score": score * lexical_weight}
        
        # Apply additional reranking factors
        for chunk_id, data in combined.items():
            chunk = data["chunk"]
            
            # Boost titles and headings
            if chunk.is_title:
                data["score"] += 0.2
            elif chunk.is_heading:
                data["score"] += 0.1
                
            # Cap at 1.0
            data["score"] = min(1.0, data["score"])
        
        # Convert to list and sort
        results = [(data["chunk"], data["score"]) for data in combined.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def keyword_search(self, keyword: str) -> List[Tuple[str, DocumentChunk]]:
        """Search for keyword matches in document chunks."""
        if not self.document_chunks:
            self.process_all_documents()
            
        if not self.document_chunks:
            logger.warning("No document chunks for keyword search")
            return []
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_id, chunks in self.document_chunks.items():
            for chunk in chunks:
                if keyword_lower in chunk.text.lower():
                    results.append((doc_id, chunk))
        
        return results
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        if not text or len(text.strip()) == 0:
            return []
            
        try:
            doc = nlp(text[:50000])  # Limit text length to avoid memory issues
            
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
            
            # Remove duplicates and sort by length (longer terms first)
            key_terms = list(set(key_terms))
            key_terms.sort(key=lambda x: len(x), reverse=True)
            
            return key_terms[:7]  # Return top 7 terms
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}")
            return []


class QueryProcessor:
    """Enhanced query processing with expansion and reformulation."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.nlp = nlp
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query to extract key concepts and expand with related terms.
        
        Returns a dictionary with:
        - original_query: The original query string
        - expanded_query: Query expanded with synonyms and related terms
        - keywords: Important keywords extracted from the query
        - entities: Named entities in the query
        - query_type: Type of query (question, command, etc.)
        - question_type: For questions, what kind of question it is
        """
        query = query.strip()
        
        # Basic processing
        result = {
            "original_query": query,
            "expanded_query": query,
            "keywords": [],
            "entities": [],
            "query_type": "unknown",
            "question_type": None
        }
        
        # Parse the query
        doc = self.nlp(query)
        
        # Extract query type
        if query.endswith('?'):
            result["query_type"] = "question"
        elif query.endswith('!'):
            result["query_type"] = "command"
        elif query.lower().startswith(('find', 'search', 'get', 'retrieve')):
            result["query_type"] = "search"
        elif query.lower().startswith(('tell', 'explain', 'describe', 'elaborate')):
            result["query_type"] = "explanation"
        else:
            result["query_type"] = "statement"
        
        # For questions, determine question type
        if result["query_type"] == "question":
            question_words = {
                'what': 'definition', 
                'how': 'process', 
                'why': 'reason', 
                'when': 'time', 
                'where': 'location', 
                'who': 'person',
                'which': 'selection'
            }
            
            first_word = doc[0].text.lower() if len(doc) > 0 else ""
            if first_word in question_words:
                result["question_type"] = question_words[first_word]
            else:
                result["question_type"] = "general"
        
        # Extract keywords (important nouns, verbs, and adjectives)
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop:
                result["keywords"].append(token.text.lower())
            elif token.pos_ in ('VERB', 'ADJ') and token.is_alpha and len(token.text) > 2 and not token.is_stop:
                result["keywords"].append(token.text.lower())
        
        # Extract named entities
        for ent in doc.ents:
            result["entities"].append((ent.text, ent.label_))
            
            # Add entity text to keywords if not already there
            if ent.text.lower() not in result["keywords"]:
                result["keywords"].append(ent.text.lower())
        
        # Expand query with synonyms and related terms
        expanded_query = self._expand_query(query, result["keywords"])
        result["expanded_query"] = expanded_query
        
        return result
    
    def _expand_query(self, original_query: str, keywords: List[str]) -> str:
        """Expand query with synonyms and related terms."""
        if not keywords:
            return original_query
            
        expansion_terms = set()
        
        # Add synonyms for important keywords
        for keyword in keywords[:3]:  # Limit to top 3 keywords to avoid over-expansion
            synonyms = self._get_synonyms(keyword)
            # Add up to 2 synonyms per keyword
            expansion_terms.update(synonyms[:2])
        
        # Remove original keywords from expansion terms
        expansion_terms = expansion_terms - set(keywords)
        
        # Limit to top 5 expansion terms
        expansion_terms = list(expansion_terms)[:5]
        
        if not expansion_terms:
            return original_query
            
        # Create expanded query
        expanded_query = original_query + " " + " ".join(expansion_terms)
        
        return expanded_query
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        # Look for synonyms in WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().lower().replace('_', ' ')
                # Only add if different from original and a single word
                if synonym != word and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def rewrite_query(self, query: str, context: List[str] = None) -> str:
        """
        Rewrite an ambiguous query to be more specific using context if available.
        
        Args:
            query: The original query
            context: Optional list of previous queries or responses for context
            
        Returns:
            Rewritten query if clarification needed, otherwise original query
        """
        # Only rewrite certain types of ambiguous queries
        if len(query.split()) > 5:
            # Longer queries are usually specific enough
            return query
            
        # Check for ambiguous pronouns without context
        doc = self.nlp(query)
        has_pronoun = any(token.pos_ == 'PRON' for token in doc)
        
        if has_pronoun and not context:
            # Ambiguous pronouns without context - leave as is and let the
            # system handle it as best it can
            return query
            
        # If we have context and pronouns, try to resolve them
        if has_pronoun and context:
            # Basic pronoun resolution using last context item
            last_context = context[-1]
            
            # Extract potential entities from the last context
            last_doc = self.nlp(last_context)
            entities = [ent.text for ent in last_doc.ents]
            
            if entities:
                # Replace common pronouns with the most recent entity
                # This is a simplistic approach - a real system would use proper coreference resolution
                replacements = {
                    'it': entities[-1],
                    'this': entities[-1],
                    'that': entities[-1],
                    'they': entities[-1],
                    'them': entities[-1],
                    'these': entities[-1],
                    'those': entities[-1]
                }
                
                words = query.split()
                for i, word in enumerate(words):
                    lower_word = word.lower()
                    if lower_word in replacements:
                        words[i] = replacements[lower_word]
                        
                return ' '.join(words)
                
        return query


class SemanticProcessor:
    """Handles semantic processing, including similarity calculations and embeddings."""
    
    def __init__(self):
        """Initialize the semantic processor with the appropriate models."""
        if sentence_transformers_available:
            # Use sentence transformers for better semantic understanding
            self.model = sentence_model
            logger.info("Using SentenceTransformer for semantic processing")
        else:
            # Fall back to TF-IDF
            self.vectorizer = TfidfVectorizer(stop_words='english')
            logger.info("Using TF-IDF for semantic processing")
    
    def get_text_embedding(self, text: str):
        """Get embedding for a text."""
        if not text or len(text.strip()) == 0:
            return None
            
        try:
            if sentence_transformers_available:
                return self.model.encode(text)
            else:
                # Use TF-IDF as fallback
                return self.vectorizer.fit_transform([text])[0]
        except Exception as e:
            logger.error(f"Error getting text embedding: {str(e)}")
            return None
    
    def get_embeddings(self, texts: List[str]):
        """Get embeddings for multiple texts."""
        if not texts:
            return []
            
        try:
            if sentence_transformers_available:
                return self.model.encode(texts)
            else:
                # Use TF-IDF as fallback
                return self.vectorizer.fit_transform(texts)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        try:
            if sentence_transformers_available:
                embedding1 = self.model.encode(text1)
                embedding2 = self.model.encode(text2)
                # Compute cosine similarity
                return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            else:
                # Use TF-IDF and cosine similarity as fallback
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate similarity matrix for a list of texts."""
        if not texts:
            return np.array([])
            
        try:
            if sentence_transformers_available:
                embeddings = self.model.encode(texts)
                # Compute pairwise cosine similarities
                similarity_matrix = np.zeros((len(texts), len(texts)))
                for i in range(len(texts)):
                    for j in range(i, len(texts)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                return similarity_matrix
            else:
                # Use TF-IDF and cosine similarity as fallback
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                return cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {str(e)}")
            return np.zeros((len(texts), len(texts)))


class ImprovedSummaryGenerator:
    """Generates concise, coherent, non-repetitive summaries from retrieved documents."""
    
    def __init__(self):
        """Initialize the improved summary generator."""
        self.language = "english"
        self.stop_words = STOPWORDS
        self.semantic_processor = SemanticProcessor()
        
        # If sumy is available, initialize its components for fallback
        if sumy_available:
            self.stemmer = Stemmer(self.language)
            self.sumy_stop_words = get_stop_words(self.language)
            
            # Initialize summarizers
            self.lexrank = LexRankSummarizer(self.stemmer)
            self.lexrank.stop_words = self.sumy_stop_words
            
            self.lsa = LsaSummarizer(self.stemmer)
            self.lsa.stop_words = self.sumy_stop_words
    
    def generate_summary(self, 
                         chunks: List[DocumentChunk], 
                         query_info: Dict[str, Any], 
                         max_length: int = 500) -> str:
        """
        Generate a concise, non-repetitive summary from document chunks relevant to the query.
        
        Args:
            chunks: List of document chunks to summarize
            query_info: Query information from QueryProcessor
            max_length: Target maximum length of summary in words
            
        Returns:
            A coherent summary of the key information relevant to the query
        """
        if not chunks:
            return "No relevant information found."
            
        try:
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Process and clean the texts
            processed_texts = self._preprocess_texts(texts)
            if not processed_texts:
                return "No relevant information found after processing."
                
            # Extract query intent and key concepts from query_info
            query_concepts = self._extract_key_concepts_from_query(query_info)
            
            # Extract information units (sentences or paragraphs)
            information_units = self._extract_information_units(processed_texts, chunks)
            
            # Score information units by relevance to query
            scored_units = self._score_by_query_relevance(information_units, query_concepts, query_info["original_query"])
            
            # Remove redundant information
            unique_units = self._remove_redundancies(scored_units)
            
            # Select top units within target length
            selected_units = self._select_within_length(unique_units, max_length)
            
            # Organize information logically
            organized_units = self._organize_information(selected_units, query_concepts)
            
            # Generate the final summary
            summary = self._generate_coherent_text(organized_units, query_info["original_query"], query_concepts)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fall back to traditional summarization if available
            if sumy_available:
                return self._generate_fallback_summary(texts, query_info["original_query"])
            else:
                return f"An error occurred while generating the summary. Please try again with a different query."
    
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
    
    def _extract_key_concepts_from_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key concepts from the query info."""
        concepts = {
            'keywords': query_info.get('keywords', []),
            'entities': query_info.get('entities', []),
            'query_type': query_info.get('query_type', 'unknown'),
            'question_type': query_info.get('question_type'),
            'original_query': query_info.get('original_query', '')
        }
        
        return concepts
    
    def _extract_information_units(self, texts: List[str], chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Extract meaningful information units from texts."""
        # Extract units along with their source chunk information
        all_units = []
        
        for i, text in enumerate(texts):
            # Get the corresponding chunk
            chunk = chunks[i] if i < len(chunks) else None
            
            # If chunk is a title or heading, keep it as a unit
            if chunk and (chunk.is_title or chunk.is_heading):
                all_units.append({
                    'text': text,
                    'chunk': chunk,
                    'is_title': chunk.is_title,
                    'is_heading': chunk.is_heading
                })
                continue
                
            # For longer texts, split by sentences
            if len(text) > 100:
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    # Skip very short sentences
                    if len(sent.split()) < 5:
                        continue
                        
                    all_units.append({
                        'text': sent,
                        'chunk': chunk,
                        'is_title': False,
                        'is_heading': False
                    })
            else:
                # For shorter texts, use as is
                all_units.append({
                    'text': text,
                    'chunk': chunk,
                    'is_title': False,
                    'is_heading': False
                })
                
        # Filter out units with little information
        filtered_units = []
        for unit in all_units:
            # Always keep titles and headings
            if unit['is_title'] or unit['is_heading']:
                filtered_units.append(unit)
                continue
                
            # Skip units that are mostly stopwords
            tokens = unit['text'].lower().split()
            if not tokens:
                continue
                
            non_stop_ratio = sum(1 for t in tokens if t not in self.stop_words) / len(tokens)
            if non_stop_ratio < 0.4:
                continue
                
            filtered_units.append(unit)
            
        return filtered_units
    
    def _score_by_query_relevance(self, 
                                  units: List[Dict[str, Any]], 
                                  query_concepts: Dict[str, Any],
                                  original_query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Score information units by relevance to the query concepts."""
        if not units:
            return []
            
        scored_units = []
        
        # Convert query keywords to a space-separated string for comparison
        query_text = ' '.join(query_concepts['keywords']) if query_concepts['keywords'] else original_query
        
        # Use the semantic processor to calculate relevance
        for unit in units:
            # Base score on semantic similarity
            similarity = self.semantic_processor.calculate_similarity(query_text, unit['text'])
            
            # Apply additional scoring based on query concepts
            bonus = 0
            
            # Boost titles and headings
            if unit['is_title']:
                bonus += 0.3
            elif unit['is_heading']:
                bonus += 0.2
            
            # Check for exact entity matches
            for entity, _ in query_concepts['entities']:
                if entity.lower() in unit['text'].lower():
                    bonus += 0.2
            
            # Check if unit contains information related to question type
            if query_concepts['question_type'] == 'definition' and any(phrase in unit['text'].lower() 
                                                                      for phrase in ['is a', 'refers to', 'defined as']):
                bonus += 0.15
            elif query_concepts['question_type'] == 'process' and any(phrase in unit['text'].lower() 
                                                                     for phrase in ['steps', 'process', 'how to']):
                bonus += 0.15
                
            # Apply bonuses (capped at 1.0)
            final_score = min(1.0, similarity + bonus)
            scored_units.append((unit, final_score))
                
        # Sort by score in descending order
        scored_units.sort(key=lambda x: x[1], reverse=True)
        return scored_units
    
    def _remove_redundancies(self, scored_units: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Remove redundant information to avoid repetition."""
        if not scored_units:
            return []
            
        # Extract units and scores
        units = [item[0]['text'] for item in scored_units]
        original_items = [item for item in scored_units]
        
        # Calculate similarity matrix between all units
        similarity_matrix = self.semantic_processor.calculate_similarity_matrix(units)
        
        # Initialize list to keep track of which units to keep
        keep_mask = np.ones(len(units), dtype=bool)
        
        # Iterate through units by score (already sorted)
        for i in range(len(units)):
            if not keep_mask[i]:
                continue  # Skip if already marked for removal
                
            # Check this unit against all subsequent (lower-scored) units
            for j in range(i + 1, len(units)):
                if not keep_mask[j]:
                    continue  # Skip if already marked for removal
                    
                # If units are too similar, remove the lower-scored one
                if similarity_matrix[i, j] > 0.7:  # Threshold for similarity
                    keep_mask[j] = False
                    
        # Create filtered list of non-redundant units with their scores
        unique_units = [original_items[i] for i in range(len(units)) if keep_mask[i]]
        return unique_units
    
    def _select_within_length(self, scored_units: List[Tuple[Dict[str, Any], float]], max_length: int) -> List[Tuple[Dict[str, Any], float]]:
        """Select top units that fit within the target length."""
        if not scored_units:
            return []
            
        current_length = 0
        selected_units = []
        
        for unit, score in scored_units:
            unit_length = len(unit['text'].split())
            if current_length + unit_length <= max_length:
                selected_units.append((unit, score))
                current_length += unit_length
            
            if current_length >= max_length:
                break
                
        return selected_units
    
    def _organize_information(self, scored_units: List[Tuple[Dict[str, Any], float]], query_concepts: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Organize selected information into a logical structure."""
        if not scored_units:
            return []
            
        # Determine organization strategy based on query
        if query_concepts['question_type'] == 'definition':
            # For definition queries, put definitive statements first
            definitions = []
            elaborations = []
            examples = []
            
            for unit, score in scored_units:
                text = unit['text'].lower()
                if unit['is_title'] or unit['is_heading']:
                    # Titles and headings go first
                    definitions.append((unit, score))
                elif any(phrase in text for phrase in ['is a', 'refers to', 'defined as', 'meaning of']):
                    definitions.append((unit, score))
                elif 'example' in text or 'instance' in text or 'such as' in text:
                    examples.append((unit, score))
                else:
                    elaborations.append((unit, score))
                    
            # Combine in logical order: definition, elaboration, examples
            return definitions + elaborations + examples
            
        elif query_concepts['question_type'] == 'process':
            # Try to organize steps in a process
            # Look for indicators of sequential steps
            units_with_markers = []
            
            for unit, score in scored_units:
                text = unit['text'].lower()
                # Look for step indicators
                has_step = bool(re.search(r'(?:^|\W)(?:step\s*\d|first|second|third|next|then|finally)(?:\W|$)', text))
                priority = 2 if unit['is_title'] else 1 if unit['is_heading'] else 0
                units_with_markers.append((unit, score, has_step, priority))
            
            # Sort: first by priority, then by step marker, then by score
            ordered_units = sorted(units_with_markers, key=lambda x: (-x[3], -x[2], -x[1]))
            return [(unit, score) for unit, score, _, _ in ordered_units]
            
        elif query_concepts['query_type'] == 'explanation':
            # For explanations, start with overview then details
            overview = []
            details = []
            
            for unit, score in scored_units:
                if unit['is_title'] or unit['is_heading']:
                    overview.append((unit, score))
                elif len(unit['text'].split()) < 20:  # Shorter sentences often give overviews
                    overview.append((unit, score))
                else:
                    details.append((unit, score))
                    
            return overview + details
            
        else:
            # For other types, start with titles/headings, then highest relevance units
            titles = []
            others = []
            
            for unit, score in scored_units:
                if unit['is_title'] or unit['is_heading']:
                    titles.append((unit, score))
                else:
                    others.append((unit, score))
                    
            # Sort non-titles by score
            others.sort(key=lambda x: x[1], reverse=True)
            
            return titles + others
    
    def _generate_coherent_text(self, 
                                organized_units: List[Tuple[Dict[str, Any], float]], 
                                query: str,
                                query_concepts: Dict[str, Any]) -> str:
        """Generate the final coherent summary text."""
        if not organized_units:
            return "No relevant information was found to answer your query."
            
        # Extract units and drop scores
        units = [unit['text'] for unit, _ in organized_units]
        
        # Start with an introduction based on query type
        introduction = self._create_introduction(query, query_concepts)
        
        # Combine units into paragraphs
        paragraphs = self._create_paragraphs(organized_units)
        
        # Add a conclusion
        conclusion = self._create_conclusion(query, query_concepts)
        
        # Combine all parts
        full_text = introduction + "\n\n"
        full_text += "\n\n".join(paragraphs)
        
        if conclusion:
            full_text += "\n\n" + conclusion
            
        return full_text
    
    def _create_introduction(self, query: str, query_concepts: Dict[str, Any]) -> str:
        """Create an introductory sentence based on the query."""
        # Extract main subject from entities if available
        subject = None
        if query_concepts['entities']:
            subject = query_concepts['entities'][0][0]
        
        # If no entities, check for keywords
        if not subject and query_concepts['keywords']:
            subject = query_concepts['keywords'][0].capitalize()
        
        # Generate introduction based on query type
        if query_concepts['query_type'] == 'question':
            if query_concepts['question_type'] == 'definition':
                if subject:
                    return f"Here's information about {subject}:"
                else:
                    return "Here's the definition you requested:"
            elif query_concepts['question_type'] == 'process':
                return "Here's how the process works:"
            elif query_concepts['question_type'] == 'reason':
                return "Here's the explanation for your question:"
            else:
                return "Here's information addressing your question:"
        elif query_concepts['query_type'] == 'explanation':
            return "Here's an explanation of the topic:"
        elif query_concepts['query_type'] == 'search':
            if subject:
                return f"Here's what I found about {subject}:"
            else:
                return "Here are the search results:"
        else:
            if subject:
                return f"Here's a summary of information about {subject}:"
            else:
                return "Here's a summary of the relevant information:"
    
    def _create_paragraphs(self, scored_units: List[Tuple[Dict[str, Any], float]]) -> List[str]:
        """Combine information units into coherent paragraphs."""
        if not scored_units:
            return []
            
        # Group by chunk to maintain document coherence when possible
        chunk_groups = {}
        for unit, _ in scored_units:
            chunk_id = unit['chunk'].doc_id if unit['chunk'] else "unknown"
            if chunk_id not in chunk_groups:
                chunk_groups[chunk_id] = []
            chunk_groups[chunk_id].append(unit)
        
        paragraphs = []
        
        # Process titles and headings first
        for chunk_id, units in chunk_groups.items():
            titles = [unit for unit in units if unit['is_title']]
            headings = [unit for unit in units if unit['is_heading']]
            
            # Add titles
            for title in titles:
                paragraphs.append(title['text'])
            
            # Add headings
            for heading in headings:
                paragraphs.append(heading['text'])
        
        # Process regular content
        for chunk_id, units in chunk_groups.items():
            regular_units = [unit for unit in units if not unit['is_title'] and not unit['is_heading']]
            
            # Skip if no regular units
            if not regular_units:
                continue
                
            # If just 1-2 units, add as separate paragraphs
            if len(regular_units) <= 2:
                for unit in regular_units:
                    paragraphs.append(unit['text'])
                continue
                
            # Otherwise, try to combine related units into coherent paragraphs
            current_paragraph = []
            
            for i, unit in enumerate(regular_units):
                if i == 0:
                    current_paragraph.append(unit['text'])
                    continue
                    
                # Check similarity with last unit in current paragraph
                last_text = current_paragraph[-1]
                
                similarity = self.semantic_processor.calculate_similarity(last_text, unit['text'])
                
                if similarity > 0.5 and len(current_paragraph) < 3:
                    # Add to current paragraph if related and paragraph not too long yet
                    current_paragraph.append(unit['text'])
                else:
                    # Start a new paragraph
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = [unit['text']]
            
            # Add the last paragraph
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                
        return paragraphs
    
    def _create_conclusion(self, query: str, query_concepts: Dict[str, Any]) -> str:
        """Create a concluding sentence if appropriate."""
        # Determine if conclusion is needed based on query type
        if query_concepts['query_type'] == 'question' and query_concepts['question_type'] in ['reason', 'process']:
            return "This summary addresses the key points based on the available information."
        elif query_concepts['query_type'] == 'explanation':
            return "This explanation covers the main aspects of the topic as requested."
        elif len(query.split()) > 10:  # Longer, more complex queries often benefit from conclusion
            return "The information above represents the most relevant content found in the knowledge base."
        
        return ""  # No conclusion for simple factual queries
        
    def _generate_fallback_summary(self, texts: List[str], query: str) -> str:
        """Generate a summary using traditional methods as fallback."""
        if not sumy_available or not texts:
            return "Could not generate a summary with the available information."
        
        # Combine texts
        combined_text = "\n\n".join(texts)
        
        try:
            # Parse the text
            parser = PlaintextParser.from_string(combined_text, Tokenizer(self.language))
            
            # Use LexRank for extractive summarization
            summary_sentences = [str(s) for s in self.lexrank(parser.document, 10)]
            
            if not summary_sentences:
                return "No relevant information could be extracted."
                
            # Format into a coherent text
            intro = "Here is a summary of the relevant information:"
            body = " ".join(summary_sentences)
            
            return f"{intro}\n\n{body}"
            
        except Exception as e:
            logger.error(f"Error generating fallback summary: {str(e)}")
            return "An error occurred while generating the summary."


class EvaluationFramework:
    """Framework for evaluating RAG system performance."""
    
    def __init__(self):
        """Initialize the evaluation framework."""
        self.rouge_evaluator = Rouge() if rouge_available else None
    
    def evaluate_response(self, 
                         query: str, 
                         response: str, 
                         retrieved_chunks: List[DocumentChunk],
                         reference_summary: str = None,
                         relevant_chunks: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG system response.
        
        Args:
            query: The original query
            response: The generated response
            retrieved_chunks: The chunks retrieved by the system
            reference_summary: Optional reference summary for comparison
            relevant_chunks: Optional list of chunk IDs known to be relevant
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "query": query,
            "metrics": {}
        }
        
        # Calculate response quality metrics
        results["metrics"]["response_length"] = len(response.split())
        
        # Calculate readability metrics
        results["metrics"]["readability"] = self._calculate_readability(response)
        
        # Calculate retrieval precision if relevant chunks are provided
        if relevant_chunks and retrieved_chunks:
            retrieved_ids = [f"{chunk.doc_id}_{chunk.chunk_id}" for chunk in retrieved_chunks]
            precision = len(set(retrieved_ids).intersection(relevant_chunks)) / len(retrieved_ids)
            recall = len(set(retrieved_ids).intersection(relevant_chunks)) / len(relevant_chunks)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results["metrics"]["retrieval_precision"] = precision
            results["metrics"]["retrieval_recall"] = recall
            results["metrics"]["retrieval_f1"] = f1
        
        # Calculate ROUGE metrics if reference summary is provided
        if reference_summary and self.rouge_evaluator:
            try:
                rouge_scores = self.rouge_evaluator.get_scores(response, reference_summary)
                
                results["metrics"]["rouge_1_f"] = rouge_scores[0]["rouge-1"]["f"]
                results["metrics"]["rouge_2_f"] = rouge_scores[0]["rouge-2"]["f"]
                results["metrics"]["rouge_l_f"] = rouge_scores[0]["rouge-l"]["f"]
            except Exception as e:
                logger.error(f"Error calculating ROUGE metrics: {str(e)}")
        
        return results
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for a text."""
        # Count sentences, words, and syllables
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if not sentences or not words:
            return {
                "flesch_reading_ease": 0,
                "avg_sentence_length": 0
            }
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Estimate syllables (simplistic approach)
        syllable_count = 0
        for word in words:
            word = word.lower()
            if len(word) <= 3:
                syllable_count += 1
            else:
                # Count vowel sequences as syllables
                vowels = "aeiouy"
                prev_is_vowel = False
                count = 0
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_is_vowel:
                        count += 1
                    prev_is_vowel = is_vowel
                
                # Adjust counts for typical patterns
                if word.endswith('e'):
                    count -= 1
                if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                    count += 1
                if count == 0:
                    count = 1
                    
                syllable_count += count
        
        # Calculate Flesch Reading Ease
        if len(words) > 0 and len(sentences) > 0:
            flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / len(words)))
        else:
            flesch = 0
            
        return {
            "flesch_reading_ease": flesch,
            "avg_sentence_length": avg_sentence_length
        }
    
    def benchmark_system(self, rag_system, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark the RAG system on a set of test queries.
        
        Args:
            rag_system: The RAG system to evaluate
            test_queries: List of dictionaries with query, relevant_chunks, and optional reference_summary
            
        Returns:
            Dictionary with aggregate benchmark results
        """
        if not test_queries:
            return {"error": "No test queries provided"}
            
        results = {
            "individual_results": [],
            "aggregate_metrics": {},
            "timing": {
                "total_time": 0,
                "avg_time": 0
            }
        }
        
        start_time = time.time()
        
        for test in test_queries:
            query = test["query"]
            
            # Process the query and measure time
            query_start = time.time()
            response = rag_system.process_query(query)
            query_time = time.time() - query_start
            
            # Evaluate the response
            eval_result = self.evaluate_response(
                query=query,
                response=response["response"],
                retrieved_chunks=response.get("retrieved_chunks", []),
                reference_summary=test.get("reference_summary"),
                relevant_chunks=test.get("relevant_chunks")
            )
            
            # Add timing information
            eval_result["time_taken"] = query_time
            
            # Add to individual results
            results["individual_results"].append(eval_result)
            
        # Calculate aggregate metrics
        results["timing"]["total_time"] = time.time() - start_time
        results["timing"]["avg_time"] = results["timing"]["total_time"] / len(test_queries)
        
        # Calculate average metrics across all queries
        metrics_keys = results["individual_results"][0]["metrics"].keys()
        for key in metrics_keys:
            values = [r["metrics"].get(key, 0) for r in results["individual_results"] if key in r["metrics"]]
            if values:
                results["aggregate_metrics"][key] = sum(values) / len(values)
            
        return results


class ConversationalRAGSystem:
    """Enhanced RAG system with conversation context support."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the RAG system."""
        logger.info(f"Initializing Conversational RAG system with knowledge base path: {knowledge_base_path}")
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_path}")
            
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.query_processor = QueryProcessor()
        self.summary_generator = ImprovedSummaryGenerator()
        self.evaluation = EvaluationFramework()
        
        # Conversation history
        self.conversation_history = []
        self.last_retrieved_chunks = []
    
    def process_query(self, query: str, 
                     consider_history: bool = True, 
                     verbose: bool = False) -> Dict[str, Any]:
        """Process a query and generate a response with conversation context."""
        logger.info(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.process_all_documents()
        
        # Use conversation history if available and enabled
        context_enhanced_query = query
        if consider_history and self.conversation_history:
            # Get context from previous interactions
            context = [item["query"] for item in self.conversation_history[-3:]]
            context.extend([item["response"] for item in self.conversation_history[-3:]])
            
            # Rewrite query considering context
            context_enhanced_query = self.query_processor.rewrite_query(query, context)
            
            if verbose and context_enhanced_query != query:
                logger.info(f"Rewrote query '{query}' to '{context_enhanced_query}' using conversation context")
        
        # Process the query to understand intent and extract key concepts
        query_info = self.query_processor.process_query(context_enhanced_query)
        
        if verbose:
            logger.info(f"Processed query info: {json.dumps(query_info, indent=2)}")
            
        # Determine query approach (hybrid search for questions, keyword search for simple lookups)
        is_question = query_info["query_type"] in ["question", "explanation"]
        
        # Retrieve relevant chunks
        if is_question:
            # Use expanded query for better retrieval
            retrieved_chunks = self.document_processor.hybrid_search(query_info["expanded_query"], top_k=7)
            chunks = [chunk for chunk, score in retrieved_chunks]
            retrieval_scores = [score for chunk, score in retrieved_chunks]
        else:
            # For keywords, try to find the most specific matches
            keywords = query_info["keywords"]
            # Use the longest keywords for better specificity
            keywords.sort(key=len, reverse=True)
            
            chunks = []
            seen_chunks = set()
            
            # Try exact matching with top keywords
            for keyword in keywords[:3]:  # Try with top 3 longest keywords
                if len(keyword) > 3:  # Only use meaningful keywords
                    matches = self.document_processor.keyword_search(keyword)
                    for doc_id, chunk in matches:
                        chunk_id = f"{doc_id}_{chunk.chunk_id}"
                        if chunk_id not in seen_chunks:
                            chunks.append(chunk)
                            seen_chunks.add(chunk_id)
            
            # If no exact matches, fall back to hybrid search
            if not chunks:
                retrieved_chunks = self.document_processor.hybrid_search(query_info["expanded_query"], top_k=7)
                chunks = [chunk for chunk, score in retrieved_chunks]
                retrieval_scores = [score for chunk, score in retrieved_chunks]
            else:
                # Assign default scores for keyword matches
                retrieval_scores = [0.9] * len(chunks)
        
        # Save retrieved chunks for context
        self.last_retrieved_chunks = chunks
        
        if not chunks:
            logger.warning(f"No relevant chunks found for query: {query}")
            response_text = "I couldn't find any relevant information to answer your query. Could you please rephrase or provide more details?"
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "response": response_text,
                "retrieved_chunks": []
            })
            
            return {
                "success": False,
                "query": query,
                "response": response_text,
                "retrieved_chunks": [],
                "retrieval_scores": []
            }
        
        logger.info(f"Found {len(chunks)} relevant chunks")
        
        # Generate concise summary
        summary = self.summary_generator.generate_summary(chunks, query_info)
        
        # Extract key terms for additional context
        all_content = " ".join([chunk.text for chunk in chunks])
        key_terms = self.document_processor.extract_key_terms(all_content)
        
        # Structure the final response
        response = self._format_response(summary, key_terms, query)
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "response": response,
            "retrieved_chunks": chunks
        })
        
        # Limit conversation history size
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "retrieved_chunks": chunks,
            "retrieval_scores": retrieval_scores
        }
    
    def _format_response(self, summary: str, key_terms: List[str], query: str) -> str:
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
    
    def generate_pdf_report(self, query: str, response: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """Generate a PDF report of the response."""
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
                    fontSize=10,
                    leading=14,
                    alignment=TA_JUSTIFY
                ))
                
                # Create elements for the PDF
                elements = []
                
                # Add title
                title_style = styles['Heading1']
                title_style.alignment = TA_LEFT
                elements.append(Paragraph(f"Summary: {query}", title_style))
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
                elements.append(Paragraph("Sources", styles['Heading2']))
                elements.append(Spacer(1, 0.1 * inch))
                
                # Group sources by document
                doc_sources = {}
                for chunk in retrieved_chunks:
                    if chunk.doc_id not in doc_sources:
                        doc_sources[chunk.doc_id] = []
                    doc_sources[chunk.doc_id].append(chunk)
                
                for doc_id, chunks in doc_sources.items():
                    elements.append(Paragraph("• " + doc_id, styles['Normal']))
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
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self, max_items: int = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        if max_items:
            return self.conversation_history[-max_items:]
        return self.conversation_history


# Main function to run the enhanced RAG system
def main():
    """Main function to run the enhanced RAG system."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing Enhanced RAG system with knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
    # Initialize the RAG system
    rag_system = ConversationalRAGSystem(KNOWLEDGE_BASE_PATH)
    
    print("\nSetup completed. You can now interact with the RAG system.")
    print("Type 'exit' to quit, 'clear' to clear conversation history, 'pdf' to generate a PDF of the last response.")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'clear':
            rag_system.clear_conversation_history()
            print("Conversation history cleared.")
            continue
        elif query.lower() == 'pdf' and rag_system.conversation_history:
            last_interaction = rag_system.conversation_history[-1]
            try:
                pdf_path = rag_system.generate_pdf_report(
                    last_interaction["query"],
                    last_interaction["response"],
                    last_interaction["retrieved_chunks"]
                )
                print(f"\nPDF report generated at: {pdf_path}")
            except Exception as e:
                print(f"\nError generating PDF report: {str(e)}")
            continue
        
        print("\nProcessing your query...\n")
        
        # Process the query
        result = rag_system.process_query(query, verbose=True)
        
        if result["success"]:
            print("=" * 80)
            print("RESPONSE:")
            print("-" * 80)
            print(result["response"])
            print("=" * 80)
            
            # Show source documents
            docs = list(set([chunk.doc_id for chunk in result["retrieved_chunks"]]))
            print("\nSources:", ", ".join(docs))
            
            # Show top 3 chunks with scores
            print("\nTop retrieved chunks:")
            for i, (chunk, score) in enumerate(zip(result["retrieved_chunks"][:3], result["retrieval_scores"][:3])):
                print(f"{i+1}. [{chunk.doc_id}] (Score: {score:.2f}): {chunk.text[:100]}...")
        else:
            print(f"Error: {result['response']}")


# For use in a Jupyter notebook
def create_rag_system():
    """Create and return a RAG system instance for use in a notebook."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    return ConversationalRAGSystem(KNOWLEDGE_BASE_PATH)


def process_query_and_generate_pdf(rag_system, query):
    """Process a query and generate a PDF report."""
    result = rag_system.process_query(query)
    
    if result["success"]:
        print("=" * 80)
        print("RESPONSE:")
        print("-" * 80)
        print(result["response"])
        print("=" * 80)
        
        # Show source documents
        docs = list(set([chunk.doc_id for chunk in result["retrieved_chunks"]]))
        print("\nSources:", ", ".join(docs))
        
        # Generate PDF report
        try:
            pdf_path = rag_system.generate_pdf_report(
                query,
                result["response"],
                result["retrieved_chunks"]
            )
            print(f"\nPDF report generated at: {pdf_path}")
            return result["response"], pdf_path
        except Exception as e:
            print(f"\nError generating PDF report: {str(e)}")
            return result["response"], None
    else:
        print(f"Error: {result['response']}")
        return None, None


def create_evaluation_dataset(sample_queries, reference_docs):
    """Create an evaluation dataset from sample queries and reference docs."""
    dataset = []
    
    for query, docs in zip(sample_queries, reference_docs):
        dataset.append({
            "query": query,
            "relevant_chunks": docs
        })
        
    return dataset


def run_evaluation(rag_system, eval_dataset):
    """Run evaluation on the RAG system."""
    evaluator = EvaluationFramework()
    results = evaluator.benchmark_system(rag_system, eval_dataset)
    
    print("Evaluation Results:")
    print("-" * 40)
    print(f"Average processing time: {results['timing']['avg_time']:.2f} seconds")
    print("\nAggregate Metrics:")
    for metric, value in results['aggregate_metrics'].items():
        print(f"{metric}: {value:.4f}")
        
    return results


# Example usage
def example_usage():
    """Show example usage of the RAG system."""
    # Create RAG system
    rag = create_rag_system()
    
    # Process a query
    query = "What are the key features of the enhanced RAG system?"
    result = rag.process_query(query)
    
    print(result["response"])
    
    # Follow-up query using conversation context
    follow_up = "What improvements does it make over traditional systems?"
    result2 = rag.process_query(follow_up)
    
    print("\nFollow-up Response:")
    print(result2["response"])
    
    # Generate PDF
    pdf_path = rag.generate_pdf_report(
        follow_up,
        result2["response"],
        result2["retrieved_chunks"]
    )
    print(f"\nPDF report generated at: {pdf_path}")


# Run the system if executed directly
if __name__ == "__main__":
    main()
