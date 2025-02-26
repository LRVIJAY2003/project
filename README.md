# project
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
        'sumy'
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

# Download required NLTK data
print("\nDownloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
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


class DocumentProcessor:
    """Handles document processing, indexing, and retrieval."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the document processor."""
        logger.info(f"Initializing DocumentProcessor with path: {knowledge_base_path}")
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1  # Set to 1 to handle cases with few documents
        )
        self.doc_embeddings = {}
    
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
    
    def create_document_embeddings(self):
        """Create embeddings for all documents."""
        if not self.documents:
            self.load_all_documents()
        
        if not self.documents:
            logger.warning("No documents loaded, cannot create embeddings")
            return
            
        docs_content = list(self.documents.values())
        doc_names = list(self.documents.keys())
        
        try:
            # Fit and transform to get document embeddings
            tfidf_matrix = self.vectorizer.fit_transform(docs_content)
            
            # Store embeddings with their document names
            for i, doc_name in enumerate(doc_names):
                self.doc_embeddings[doc_name] = tfidf_matrix[i]
                
        except Exception as e:
            logger.error(f"Error creating document embeddings: {str(e)}")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Search for documents relevant to the query."""
        if not self.doc_embeddings:
            self.create_document_embeddings()
            
        if not self.doc_embeddings:
            logger.warning("No document embeddings available for search")
            return []
            
        # Transform query using the same vectorizer
        try:
            query_vector = self.vectorizer.transform([query.lower()])
            
            results = []
            
            # Calculate similarity between query and all documents
            for doc_name, doc_vector in self.doc_embeddings.items():
                similarity = cosine_similarity(query_vector, doc_vector)[0][0]
                results.append((doc_name, similarity, self.documents[doc_name]))
            
            # Sort by similarity (highest first) and return top_k results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """Search for keyword matches in documents."""
        if not self.documents:
            self.load_all_documents()
            
        if not self.documents:
            logger.warning("No documents loaded for keyword search")
            return []
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            if keyword_lower in content.lower():
                # Extract context around the keyword
                context = self.extract_context(content, keyword_lower)
                if context:
                    results.append((doc_name, context))
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 500) -> str:
        """Extract context around a keyword in text."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(re.escape(keyword_lower), text_lower)]
        
        if not matches:
            return ""
        
        # Extract context around first occurrence
        start_pos = max(0, matches[0] - context_size)
        end_pos = min(len(text), matches[0] + len(keyword) + context_size)
        
        # Find sentence boundaries
        if start_pos > 0:
            sentence_start = text.rfind('.', 0, start_pos)
            if sentence_start != -1:
                start_pos = sentence_start + 1
        
        if end_pos < len(text):
            sentence_end = text.find('.', end_pos)
            if sentence_end != -1:
                end_pos = sentence_end + 1
        
        return text[start_pos:end_pos].strip()
    
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
    
    def generate_summary(self, texts: List[str], query: str, max_length: int = 500) -> str:
        """
        Generate a concise, non-repetitive summary from documents relevant to the query.
        
        Args:
            texts: List of document texts to summarize
            query: The user's query
            max_length: Target maximum length of summary in words
            
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
            
            # Extract information units (sentences or paragraphs)
            information_units = self._extract_information_units(processed_texts)
            
            # Score information units by relevance to query
            scored_units = self._score_by_query_relevance(information_units, query_concepts)
            
            # Remove redundant information
            unique_units = self._remove_redundancies(scored_units)
            
            # Select top units within target length
            selected_units = self._select_within_length(unique_units, max_length)
            
            # Organize information logically
            organized_units = self._organize_information(selected_units, query_concepts)
            
            # Generate the final summary
            summary = self._generate_coherent_text(organized_units, query)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fall back to traditional summarization if available
            if sumy_available:
                return self._generate_fallback_summary(texts, query)
            else:
                return f"An error occurred while generating the summary. Please try again with a different query."
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Clean and normalize the input texts."""
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                continue
                
            # Normalize whitespac