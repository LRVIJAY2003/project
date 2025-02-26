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
    
    def _extract_information_units(self, texts: List[str]) -> List[str]:
        """Extract meaningful information units from texts."""
        # For longer documents, use paragraphs as units
        # For shorter texts, use sentences
        all_units = []
        
        for text in texts:
            # If text is long, split by paragraphs first
            if len(text) > 1000:
                paragraphs = [p for p in text.split('\n\n') if p.strip()]
                
                # If paragraphs are too long, split them into sentences
                for para in paragraphs:
                    if len(para) > 300:
                        sentences = nltk.sent_tokenize(para)
                        all_units.extend(sentences)
                    else:
                        all_units.append(para)
            else:
                # For shorter texts, use sentences
                sentences = nltk.sent_tokenize(text)
                all_units.extend(sentences)
                
        # Filter out very short units or units with little information
        filtered_units = []
        for unit in all_units:
            # Skip very short units
            if len(unit.split()) < 5:
                continue
                
            # Skip units that are mostly stopwords
            tokens = unit.lower().split()
            non_stop_ratio = sum(1 for t in tokens if t not in self.stop_words) / len(tokens)
            if non_stop_ratio < 0.4:
                continue
                
            filtered_units.append(unit)
            
        return filtered_units
    
    def _score_by_query_relevance(self, units: List[str], query_concepts: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Score information units by relevance to the query concepts."""
        if not units:
            return []
            
        scored_units = []
        
        # Convert query keywords to a space-separated string for comparison
        query_text = ' '.join(query_concepts['keywords']) if query_concepts['keywords'] else query_concepts['original_query']
        
        # Use the semantic processor to calculate relevance
        for unit in units:
            # Base score on semantic similarity
            similarity = self.semantic_processor.calculate_similarity(query_text, unit)
            
            # Apply additional scoring based on query concepts
            bonus = 0
            
            # Check for exact entity matches
            for entity, _ in query_concepts['entities']:
                if entity.lower() in unit.lower():
                    bonus += 0.2
            
            # Check if unit contains information related to question type
            if query_concepts['question_type'] == 'definition' and any(phrase in unit.lower() 
                                                                    for phrase in ['is a', 'refers to', 'defined as']):
                bonus += 0.15
            elif query_concepts['question_type'] == 'process' and any(phrase in unit.lower() 
                                                                    for phrase in ['steps', 'process', 'how to']):
                bonus += 0.15
                
            # Apply bonuses (capped at 1.0)
            final_score = min(1.0, similarity + bonus)
            scored_units.append((unit, final_score))
                
        # Sort by score in descending order
        scored_units.sort(key=lambda x: x[1], reverse=True)
        return scored_units
    
    def _remove_redundancies(self, scored_units: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Remove redundant information to avoid repetition."""
        if not scored_units:
            return []
            
        # Extract units and scores
        units = [item[0] for item in scored_units]
        scores = [item[1] for item in scored_units]
        
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
        unique_units = [(units[i], scores[i]) for i in range(len(units)) if keep_mask[i]]
        return unique_units
    
    def _select_within_length(self, scored_units: List[Tuple[str, float]], max_length: int) -> List[Tuple[str, float]]:
        """Select top units that fit within the target length."""
        if not scored_units:
            return []
            
        current_length = 0
        selected_units = []
        
        for unit, score in scored_units:
            unit_length = len(unit.split())
            if current_length + unit_length <= max_length:
                selected_units.append((unit, score))
                current_length += unit_length
            
            if current_length >= max_length:
                break
                
        return selected_units
    
    def _organize_information(self, scored_units: List[Tuple[str, float]], query_concepts: Dict[str, Any]) -> List[Tuple[str, float]]:
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
                lower_unit = unit.lower()
                if any(phrase in lower_unit for phrase in ['is a', 'refers to', 'defined as', 'meaning of']):
                    definitions.append((unit, score))
                elif 'example' in lower_unit or 'instance' in lower_unit:
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
                # Look for step indicators
                has_step = bool(re.search(r'(?:^|\W)(?:step\s*\d|first|second|third|next|then|finally)(?:\W|$)', unit.lower()))
                units_with_markers.append((unit, score, has_step))
            
            # Sort: first steps with markers, then by score
            ordered_units = sorted(units_with_markers, key=lambda x: (-x[2], -x[1]))
            return [(unit, score) for unit, score, _ in ordered_units]
            
        else:
            # For other types, start with highest relevance units
            return scored_units
    
    def _generate_coherent_text(self, organized_units: List[Tuple[str, float]], query: str) -> str:
        """Generate the final coherent summary text."""
        if not organized_units:
            return "No relevant information was found to answer your query."
            
        # Extract units and drop scores
        units = [unit for unit, _ in organized_units]
        
        # Start with an introduction
        introduction = self._create_introduction(query)
        
        # Combine units into paragraphs
        paragraphs = self._create_paragraphs(units)
        
        # Add a conclusion
        conclusion = self._create_conclusion(query)
        
        # Combine all parts
        full_text = introduction + "\n\n"
        full_text += "\n\n".join(paragraphs)
        
        if conclusion:
            full_text += "\n\n" + conclusion
            
        return full_text
    
    def _create_introduction(self, query: str) -> str:
        """Create an introductory sentence based on the query."""
        # Parse the query
        doc = nlp(query)
        
        # Extract main subject of the query
        subject = None
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj', 'pobj') and token.pos_ in ('NOUN', 'PROPN'):
                subject = token.text
                break
                
        if not subject:
            # Look for any noun phrase if no subject found
            for token in doc:
                if token.pos_ in ('NOUN', 'PROPN'):
                    subject = token.text
                    break
        
        # Generate introduction
        if subject:
            return f"Here is a summary of information about {subject}:"
        else:
            return "Here is a summary of the relevant information:"
    
    def _create_paragraphs(self, units: List[str]) -> List[str]:
        """Combine information units into coherent paragraphs."""
        if not units:
            return []
            
        # If we have just a few units, make each its own paragraph
        if len(units) <= 3:
            return units
            
        # Otherwise, try to combine related units
        paragraphs = []
        current_paragraph = [units[0]]
        
        for i in range(1, len(units)):
            current_unit = units[i]
            prev_unit = units[i-1]
            
            # Simple heuristic: if units share significant words, they might be related
            current_words = set(current_unit.lower().split())
            prev_words = set(prev_unit.lower().split())
            
            # Remove stopwords
            current_words = {w for w in current_words if w not in self.stop_words}
            prev_words = {w for w in prev_words if w not in self.stop_words}
            
            # Check overlap
            overlap = len(current_words.intersection(prev_words))
            
            if overlap >= 3 and len(current_paragraph) < 3:
                # Add to current paragraph if related and paragraph not too long yet
                current_paragraph.append(current_unit)
            else:
                # Start a new paragraph
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = [current_unit]
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
            
        return paragraphs
    
    def _create_conclusion(self, query: str) -> str:
        """Create a concluding sentence if appropriate."""
        # Parse query to see if it's a type that benefits from a conclusion
        doc = nlp(query)
        
        if len(doc) > 0:
            first_word = doc[0].text.lower()
            
            # Questions about reasons or explanations often benefit from conclusions
            if first_word in ['why', 'how'] or 'explain' in query.lower():
                return "These are the key points addressing your query based on the available information."
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


class RAGSystem:
    """Retrieval-Augmented Generation system for document Q&A."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the RAG system."""
        logger.info(f"Initializing RAG system with knowledge base path: {knowledge_base_path}")
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_path}")
            
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.summary_generator = ImprovedSummaryGenerator()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and generate a response."""
        logger.info(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.load_all_documents()
        
        # Determine query type
        is_question = self._is_question(query)
        logger.info(f"Query identified as {'question' if is_question else 'keyword search'}")
        
        # Retrieve relevant documents
        if is_question:
            retrieved_docs = self.document_processor.search_documents(query, top_k=5)
            documents = [(doc_name, content) for doc_name, score, content in retrieved_docs]
        else:
            # For keywords, try exact matching
            keywords = query.split()
    

class RAGSystem:
    """Retrieval-Augmented Generation system for document Q&A."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the RAG system."""
        logger.info(f"Initializing RAG system with knowledge base path: {knowledge_base_path}")
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_path}")
            
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.summary_generator = ImprovedSummaryGenerator()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and generate a response."""
        logger.info(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.load_all_documents()
        
        # Determine query type
        is_question = self._is_question(query)
        logger.info(f"Query identified as {'question' if is_question else 'keyword search'}")
        
        # Retrieve relevant documents
        if is_question:
            retrieved_docs = self.document_processor.search_documents(query, top_k=5)
            documents = [(doc_name, content) for doc_name, score, content in retrieved_docs]
        else:
            # For keywords, try exact matching
            keywords = query.split()
            # Use the longest keyword for exact matches
            keywords.sort(key=len, reverse=True)
            
            documents = []
            for keyword in keywords[:3]:  # Try with top 3 longest keywords
                if len(keyword) > 3:  # Only use meaningful keywords
                    matches = self.document_processor.keyword_search(keyword)
                    for doc_name, content in matches:
                        if doc_name not in [d for d, _ in documents]:
                            documents.append((doc_name, content))
            
            # If no exact matches, fall back to semantic search
            if not documents:
                retrieved_docs = self.document_processor.search_documents(query, top_k=5)
                documents = [(doc_name, content) for doc_name, score, content in retrieved_docs]
        
        if not documents:
            logger.warning(f"No relevant documents found for query: {query}")
            return {
                "success": False,
                "error": "No relevant information found.",
                "query": query,
                "retrieved_docs": []
            }
        
        logger.info(f"Found {len(documents)} relevant documents")
        
        # Extract texts for summarization
        texts = [content for _, content in documents]
        
        # Generate concise summary
        summary = self.summary_generator.generate_summary(texts, query)
        
        # Extract key terms for additional context
        all_content = " ".join(texts)
        key_terms = self.document_processor.extract_key_terms(all_content)
        
        # Structure the final response
        response = self._format_response(summary, key_terms, query)
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "retrieved_docs": [doc_name for doc_name, _ in documents]
        }
    
    def _is_question(self, text: str) -> bool:
        """Determine if the text is a question."""
        # Check for question marks
        if '?' in text:
            return True
        
        # Check for question words
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'tell', 'explain', 'describe']
        first_word = text.lower().split()[0] if text else ""
        
        return first_word in question_starters
    
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
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
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
                
                for doc in retrieved_docs:
                    elements.append(Paragraph("• " + doc, styles['Normal']))
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


# Main function to run the RAG system
def main():
    """Main function to run the RAG system."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing RAG system with knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
    # Initialize the RAG system
    rag_system = RAGSystem(KNOWLEDGE_BASE_PATH)
    
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
            print("\nSources:", ", ".join(result["retrieved_docs"]))
            
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
            print(f"Error: {result['error']}")


# For use in a Jupyter notebook
def create_rag_system():
    """Create and return a RAG system instance for use in a notebook."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    return RAGSystem(KNOWLEDGE_BASE_PATH)


def process_query_and_generate_pdf(rag_system, query):
    """Process a query and generate a PDF report."""
    result = rag_system.process_query(query)
    
    if result["success"]:
        print("=" * 80)
        print("RESPONSE:")
        print("-" * 80)
        print(result["response"])
        print("=" * 80)
        print("\nSources:", ", ".join(result["retrieved_docs"]))
        
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
        print(f"Error: {result['error']}")
        return None, None


# Run the system if executed directly
if __name__ == "__main__":
    main()        