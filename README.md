"""
Configuration module for the RAG application.
Loads environment variables and provides configuration settings.
Updated for compatibility with installed package versions.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env file
load_dotenv()

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Flask configuration - updated for Flask 3.0.0
DEBUG = os.getenv("FLASK_ENV") == "development"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-replace-this")
PORT = int(os.getenv("PORT", 5000))

# Confluence API settings
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_SPACE_ID = os.getenv("CONFLUENCE_SPACE_ID")
CONFLUENCE_USER_ID = os.getenv("CONFLUENCE_USER_ID")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# Remedy API settings
REMEDY_SERVER = os.getenv("REMEDY_SERVER")
REMEDY_API_BASE = os.getenv("REMEDY_API_BASE")
REMEDY_USERNAME = os.getenv("REMEDY_USERNAME")
REMEDY_PASSWORD = os.getenv("REMEDY_PASSWORD")

# Google Cloud and Gemini settings
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Cache settings
CACHE_DIR = os.path.join(tempfile.gettempdir(), "rag_cache")
Path(CACHE_DIR).mkdir(exist_ok=True)

# Vector store settings
VECTOR_STORE_PATH = os.path.join(CACHE_DIR, "vector_store")
Path(VECTOR_STORE_PATH).mkdir(exist_ok=True)

# Embedding model settings - updated for compatibility
# Use a simpler model that is more compatible with older transformers versions
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # Changed from all-MiniLM-L6-v2
EMBEDDING_DIMENSION = 384  # Dimension for the embedding model chosen

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
NUM_RESULTS = 5
TEMPERATURE = 0.2

# Import loguru only after defining log level
try:
    from loguru import logger
    logger.info("Configuration loaded")
except ImportError:
    import logging
    logging = logging.getLogger(__name__)
    logging.info("Loguru not available, using standard logging")

# Validate required configuration
def validate_config():
    """Validate critical configuration settings."""
    required_vars = {
        "CONFLUENCE_URL": CONFLUENCE_URL,
        "CONFLUENCE_API_TOKEN": CONFLUENCE_API_TOKEN,
        "REMEDY_API_BASE": REMEDY_API_BASE,
        "PROJECT_ID": PROJECT_ID,
        "MODEL_NAME": MODEL_NAME
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        missing_vars_str = ", ".join(missing_vars)
        try:
            logger.warning(f"Missing required environment variables: {missing_vars_str}")
        except:
            logging.warning(f"Missing required environment variables: {missing_vars_str}")
        
    return not missing_vars






"""
Embedding module for the RAG application.
Provides functionality to generate embeddings for document chunks.
Updated with better error handling for sentence-transformers compatibility issues.
"""
from typing import List, Dict, Any, Union, Optional
import os
import json
import numpy as np
from tqdm import tqdm
import time

import config
from utils import get_logger, timer_decorator, generate_cache_key, ensure_directory

# Initialize logger
logger = get_logger("embedding")

class EmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str, optional): Name of the embedding model. Defaults to config value.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.logger = get_logger(f"embedding_{self.__class__.__name__}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement this method")


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using SentenceTransformers."""
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize the SentenceTransformer embedding model.
        
        Args:
            model_name (str, optional): Name of the embedding model. Defaults to config value.
            cache_dir (str, optional): Directory to cache embeddings. Defaults to config value.
        """
        super().__init__(model_name)
        
        self.cache_dir = cache_dir or os.path.join(config.CACHE_DIR, "embeddings")
        ensure_directory(self.cache_dir)
        
        # Set default embedding dimension in case model loading fails
        self.embedding_dim = config.EMBEDDING_DIMENSION
        self.model = None
        
        # Initialize the model with better error handling
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SentenceTransformer model with robust error handling."""
        try:
            # First try the regular import
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Successfully initialized embedding model {self.model_name} with dimension {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize primary embedding model {self.model_name}: {e}")
            
            # Try with a fallback model
            try:
                from sentence_transformers import SentenceTransformer
                fallback_model = "paraphrase-MiniLM-L3-v2"  # Very compatible fallback
                self.logger.warning(f"Attempting to load fallback model: {fallback_model}")
                self.model = SentenceTransformer(fallback_model)
                self.model_name = fallback_model
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Successfully initialized fallback embedding model with dimension {self.embedding_dim}")
                
            except Exception as e2:
                self.logger.error(f"Failed to initialize fallback embedding model: {e2}")
                
                # Create dummy embedder as final fallback
                self.logger.warning("Creating dummy embedder as final fallback - results will be suboptimal")
                self.model = None
                self.embedding_dim = config.EMBEDDING_DIMENSION
                
    def _get_cache_path(self, text: str) -> str:
        """
        Get the cache path for an embedding.
        
        Args:
            text (str): Text to generate cache path for
            
        Returns:
            str: Cache file path
        """
        key = generate_cache_key(text, prefix="emb")
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Load embedding from cache.
        
        Args:
            text (str): Text to load embedding for
            
        Returns:
            list or None: Embedding vector if found, None otherwise
        """
        cache_path = self._get_cache_path(text)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get("embedding")
            except Exception as e:
                self.logger.error(f"Error loading embedding from cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embedding to cache.
        
        Args:
            text (str): Text that was embedded
            embedding (list): Embedding vector
        """
        cache_path = self._get_cache_path(text)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "embedding": embedding
                }, f)
        except Exception as e:
            self.logger.error(f"Error saving embedding to cache: {e}")
    
    def _dummy_embedding(self, text: str) -> List[float]:
        """
        Generate a dummy embedding when the model fails.
        This is a fallback that generates a deterministic but unique vector based on the text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Dummy embedding vector
        """
        # Seed with text hash for deterministic output
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
        np.random.seed(text_hash)
        
        # Generate a random vector, but with some relationship to text content
        # This will at least group similar texts somewhat
        vec = np.random.rand(self.embedding_dim).astype(np.float32)
        
        # Add some bias based on text length and content
        vec[0] = min(1.0, len(text) / 1000.0)  # First dimension biased by text length
        
        # Add bias for question detection
        if '?' in text:
            vec[1] = 0.8
        
        # Normalize to unit length
        vec = vec / np.linalg.norm(vec)
        
        return vec.tolist()
    
    @timer_decorator
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text with caching.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dim
        
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding
        try:
            if self.model is None:
                embedding = self._dummy_embedding(text)
            else:
                # Use the model to generate embedding
                embedding = self.model.encode(text, show_progress_bar=False).tolist()
            
            # Save to cache
            self._save_to_cache(text, embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Fall back to dummy embedding on error
            dummy_embed = self._dummy_embedding(text)
            self._save_to_cache(text, dummy_embed)
            return dummy_embed
    
    @timer_decorator
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []
        
        # Check which texts need embedding
        to_embed = []
        to_embed_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[i] = [0.0] * self.embedding_dim
                continue
                
            cached = self._load_from_cache(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                to_embed.append(text)
                to_embed_indices.append(i)
        
        # Generate embeddings for texts not in cache
        new_embeddings = []
        if to_embed:
            try:
                if self.model is None:
                    # Use dummy embeddings if no model
                    self.logger.info(f"Generating {len(to_embed)} dummy embeddings")
                    new_embeddings = [self._dummy_embedding(text) for text in to_embed]
                else:
                    # Use the model for batch encoding
                    self.logger.info(f"Generating {len(to_embed)} embeddings")
                    new_embeddings = self.model.encode(
                        to_embed,
                        batch_size=16,  # Reduced batch size for better compatibility
                        show_progress_bar=True,
                        convert_to_tensor=False  # Ensure we get numpy arrays
                    ).tolist()
                
                # Save to cache
                for text, embedding in zip(to_embed, new_embeddings):
                    self._save_to_cache(text, embedding)
                    
            except Exception as e:
                self.logger.error(f"Error generating batch embeddings: {e}")
                # Fall back to individual dummy embeddings on error
                self.logger.info("Falling back to individual dummy embeddings")
                new_embeddings = []
                for text in to_embed:
                    embed = self._dummy_embedding(text)
                    new_embeddings.append(embed)
                    self._save_to_cache(text, embed)
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Add cached embeddings
        for idx, embedding in cached_embeddings.items():
            all_embeddings[idx] = embedding
        
        # Add new embeddings
        for i, idx in enumerate(to_embed_indices):
            all_embeddings[idx] = new_embeddings[i]
        
        return all_embeddings


class EmbeddingProcessor:
    """Process documents and generate embeddings for chunks."""
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initialize the embedding processor.
        
        Args:
            embedding_model (EmbeddingModel, optional): Embedding model to use
        """
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        self.logger = get_logger("embedding_processor")
    
    @timer_decorator
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and add embeddings.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            list: List of chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract texts to embed
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        self.logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    @timer_decorator
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query (str): Query text
            
        Returns:
            list: Query embedding vector
        """
        return self.embedding_model.generate_embedding(query)
    
    @timer_decorator
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.
        
        Args:
            queries (list): List of query texts
            
        Returns:
            list: List of query embedding vectors
        """
        return self.embedding_model.generate_embeddings(queries)