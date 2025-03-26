"""
Embedding module for the RAG application.
Provides functionality to generate embeddings for document chunks.
Updated for compatibility with sentence-transformers 2.4.1
"""
from typing import List, Dict, Any, Union, Optional
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
        
        # Initialize the model - updated for compatibility with newer versions
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension - compatible with sentence-transformers 2.4.1
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Initialized embedding model {self.model_name} with dimension {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            raise
    
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
            # Updated to handle newer sentence-transformers API
            embedding = self.model.encode(text, show_progress_bar=False).tolist()
            
            # Save to cache
            self._save_to_cache(text, embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return zero vector on error
            return [0.0] * self.embedding_dim
    
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
                self.logger.info(f"Generating {len(to_embed)} embeddings")
                # Updated for newer sentence-transformers API
                new_embeddings = self.model.encode(
                    to_embed,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_tensor=False  # Ensure we get numpy arrays
                ).tolist()
                
                # Save to cache
                for text, embedding in zip(to_embed, new_embeddings):
                    self._save_to_cache(text, embedding)
            except Exception as e:
                self.logger.error(f"Error generating batch embeddings: {e}")
                # Return zero vectors on error
                new_embeddings = [[0.0] * self.embedding_dim for _ in range(len(to_embed))]
        
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






"""
Google Gemini integration for the RAG application.
Updated for compatibility with google-cloud-aiplatform 1.36.0
"""
from typing import List, Dict, Any, Optional, Union, Generator
import os
import time
import json

from google.api_core.exceptions import GoogleAPIError
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config
from utils import get_logger, timer_decorator, format_chunk_for_prompt

# Initialize logger
logger = get_logger("gemini")

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize the Gemini client.
        
        Args:
            model_name (str, optional): Gemini model name. Defaults to config value.
            temperature (float, optional): Temperature for generation. Defaults to config value.
        """
        self.model_name = model_name or config.MODEL_NAME
        self.temperature = temperature or config.TEMPERATURE
        self.logger = get_logger("gemini_client")
        
        # Initialize Vertex AI with updated API for 1.36.0
        try:
            # Initialize API with project and location
            vertexai.init(project=config.PROJECT_ID, location=config.REGION)
            self.logger.info(f"Initialized Vertex AI for project {config.PROJECT_ID} in {config.REGION}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
        
        # Initialize model - compatible with latest google-cloud-aiplatform
        try:
            self.model = GenerativeModel(self.model_name)
            self.logger.info(f"Initialized Gemini model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model {self.model_name}: {e}")
            raise
    
    @retry(
        retry=retry_if_exception_type(GoogleAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @timer_decorator
    def generate_text(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate text response from Gemini.
        
        Args:
            prompt (str): Prompt text
            temperature (float, optional): Temperature for generation. Defaults to instance value.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 1024.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            str or Generator: Generated text or response stream
        """
        if temperature is None:
            temperature = self.temperature
        
        try:
            # Create content - updated for latest API
            content = [Content.from_str(prompt)]
            
            # Generate response with updated parameter names for newer API
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
            
            if stream:
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config,
                    stream=True
                )
                
                # Return the streaming response for the caller to process
                return response
            else:
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config
                )
                
                # Extract text from response - compatible with latest API
                return response.text
                
        except GoogleAPIError as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    @retry(
        retry=retry_if_exception_type(GoogleAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @timer_decorator
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate response in a chat format from Gemini.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            temperature (float, optional): Temperature for generation. Defaults to instance value.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 1024.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            str or Generator: Generated text or response stream
        """
        if temperature is None:
            temperature = self.temperature
        
        try:
            # Convert messages to Gemini format - updated for current API
            content_parts = []
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    content_parts.append(Content(role="user", parts=[Part.from_text(content)]))
                elif role in ["assistant", "model"]:
                    content_parts.append(Content(role="model", parts=[Part.from_text(content)]))
                elif role == "system":
                    # For system messages in newer API versions
                    content_parts.append(Content(role="user", parts=[Part.from_text(f"[SYSTEM] {content}")]))
            
            # Generate response with updated parameter names
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
            
            if stream:
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    stream=True
                )
                
                # Return the streaming response for the caller to process
                return response
            else:
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config
                )
                
                # Extract text from response
                return response.text
                
        except GoogleAPIError as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            return f"Error generating response: {str(e)}"


class RAGGenerator:
    """Generate responses using the RAG approach with Gemini."""
    
    def __init__(self, gemini_client: GeminiClient = None):
        """
        Initialize the RAG generator.
        
        Args:
            gemini_client (GeminiClient, optional): Gemini client to use
        """
        self.gemini_client = gemini_client or GeminiClient()
        self.logger = get_logger("rag_generator")
    
    @timer_decorator
    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate a response using RAG.
        
        Args:
            query (str): User query
            chunks (list): Retrieved document chunks
            system_prompt (str, optional): System prompt for the LLM
            temperature (float, optional): Temperature for generation
            max_tokens (int, optional): Maximum tokens to generate
            stream (bool, optional): Whether to stream the response
            
        Returns:
            str or Generator: Generated text or response stream
        """
        # Format chunks
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            
            # Format the chunk for inclusion in the prompt
            formatted_chunk = f"[Document {i+1}] {title}\n"
            formatted_chunk += f"Source: {source}\n"
            if url:
                formatted_chunk += f"URL: {url}\n"
            formatted_chunk += f"Content: {text}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        # Default system prompt if not provided
        if not system_prompt:
            system_prompt = """
            You are an intelligent assistant that provides helpful, accurate, 
            and thoughtful responses to queries. Base your answers on the 
            provided context documents. If the documents don't provide enough 
            information to answer completely, acknowledge the limitations of your 
            knowledge. Always cite the sources of information in your response.
            """.strip()
        
        # Build the full prompt
        prompt = f"""
        {system_prompt}
        
        Here are the reference documents:
        
        {"\n\n".join(formatted_chunks)}
        
        User Query: {query}
        
        Please provide a comprehensive answer based on the provided documents.
        If the documents don't provide enough information, say so clearly.
        Include citations to the relevant document numbers in your response.
        """.strip()
        
        # Generate response
        start_time = time.time()
        response = self.gemini_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if not stream:
            generation_time = time.time() - start_time
            response_length = len(response) if isinstance(response, str) else "streaming"
            self.logger.info(f"Generated RAG response ({response_length} chars) in {generation_time:.2f} seconds")
        
        return response
    
    @timer_decorator
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        chunks: List[Dict[str, Any]],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate a chat response using RAG.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            chunks (list): Retrieved document chunks
            system_prompt (str, optional): System prompt for the LLM
            temperature (float, optional): Temperature for generation
            max_tokens (int, optional): Maximum tokens to generate
            stream (bool, optional): Whether to stream the response
            
        Returns:
            str or Generator: Generated text or response stream
        """
        # Format chunks
        context_str = ""
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            
            # Format the chunk for inclusion in the prompt
            chunk_str = f"[Document {i+1}] {title}\n"
            chunk_str += f"Source: {source}\n"
            if url:
                chunk_str += f"URL: {url}\n"
            chunk_str += f"Content: {text}\n\n"
            
            context_str += chunk_str
        
        # Default system prompt if not provided
        if not system_prompt:
            system_prompt = """
            You are an intelligent assistant that provides helpful, accurate, 
            and thoughtful responses to queries. Base your answers on the 
            provided context documents. If the documents don't provide enough 
            information to answer completely, acknowledge the limitations of your 
            knowledge. Always cite the sources of information in your response.
            """.strip()
        
        # Create a new message list with context and system prompt
        new_messages = []
        
        # Add system prompt and context as the first message
        system_with_context = f"""
        {system_prompt}
        
        Here are the reference documents:
        
        {context_str}
        
        Please provide answers based on the provided documents.
        If the documents don't provide enough information, say so clearly.
        Include citations to the relevant document numbers in your response.
        """.strip()
        
        new_messages.append({"role": "system", "content": system_with_context})
        
        # Add user messages
        for message in messages:
            new_messages.append(message)
        
        # Generate response
        start_time = time.time()
        response = self.gemini_client.chat(
            messages=new_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if not stream:
            generation_time = time.time() - start_time
            response_length = len(response) if isinstance(response, str) else "streaming"
            self.logger.info(f"Generated RAG chat response ({response_length} chars) in {generation_time:.2f} seconds")
        
        return response








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

# Embedding model settings - updated for sentence-transformers 2.4.1
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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
Indexing module for the RAG application.
Provides functionality to build and manage vector indexes for document retrieval.
Updated for compatibility with faiss-cpu 1.7.4
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from utils import get_logger, timer_decorator, ensure_directory

# Initialize logger
logger = get_logger("indexing")

class VectorIndex:
    """Base class for vector indexes."""
    
    def __init__(self, index_dir: str = None, dimension: int = None):
        """
        Initialize the vector index.
        
        Args:
            index_dir (str, optional): Directory to store the index. Defaults to config value.
            dimension (int, optional): Dimension of the vectors. Defaults to config value.
        """
        self.index_dir = index_dir or config.VECTOR_STORE_PATH
        self.dimension = dimension or config.EMBEDDING_DIMENSION
        self.logger = get_logger(f"vector_index_{self.__class__.__name__}")
        
        # Ensure index directory exists
        ensure_directory(self.index_dir)
    
    def add(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            ids (list): List of document IDs
            vectors (list): List of embedding vectors
            metadatas (list, optional): List of document metadata
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (list): Query embedding vector
            k (int, optional): Number of results to return
            
        Returns:
            list: List of search results with IDs, scores, and metadata
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self) -> None:
        """Save the index to disk."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear(self) -> None:
        """Clear the index."""
        raise NotImplementedError("Subclasses must implement this method")


class FAISSIndex(VectorIndex):
    """Vector index implementation using FAISS."""
    
    def __init__(self, index_dir: str = None, dimension: int = None, index_type: str = "flat"):
        """
        Initialize the FAISS index.
        
        Args:
            index_dir (str, optional): Directory to store the index. Defaults to config value.
            dimension (int, optional): Dimension of the vectors. Defaults to config value.
            index_type (str, optional): Type of FAISS index ('flat', 'ivf', 'hnsw'). Defaults to 'flat'.
        """
        super().__init__(index_dir, dimension)
        
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Map FAISS indices to document IDs
        self.metadata_map = {}  # Map document IDs to metadata
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the FAISS index based on the specified type."""
        try:
            if self.index_type == "ivf":
                # IVF index - faster search, less accurate
                # Requires vectors for clustering
                nlist = max(1, min(2048, int(1000000 / self.dimension)))
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                # Note: This index requires training before adding vectors
                self.index_needs_training = True
            elif self.index_type == "hnsw":
                # HNSW index - good balance of speed and accuracy
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
                self.index_needs_training = False
            else:
                # Flat index - slower search, most accurate
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index_needs_training = False
                
            self.logger.info(f"Initialized FAISS index of type {self.index_type} with dimension {self.dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            # Fall back to flat index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_needs_training = False
            self.index_type = "flat"
    
    def _convert_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        """
        Convert list of vectors to numpy array.
        
        Args:
            vectors (list): List of embedding vectors
            
        Returns:
            numpy.ndarray: Array of vectors
        """
        # Updated for numpy 1.24.4
        return np.array(vectors, dtype=np.float32)
    
    def _train_if_needed(self, vectors: np.ndarray) -> None:
        """
        Train the index if required.
        
        Args:
            vectors (numpy.ndarray): Vectors to train on
        """
        if self.index_needs_training and not self.index.is_trained:
            if vectors.shape[0] < 100:
                self.logger.warning("Too few vectors for reliable IVF training, using random vectors")
                # Generate random vectors for training
                random_vectors = np.random.random((max(100, vectors.shape[0] * 2), self.dimension)).astype('float32')
                self.index.train(random_vectors)
            else:
                self.logger.info(f"Training IVF index with {vectors.shape[0]} vectors")
                self.index.train(vectors)
    
    @timer_decorator
    def add(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            ids (list): List of document IDs
            vectors (list): List of embedding vectors
            metadatas (list, optional): List of document metadata
        """
        if not ids or not vectors:
            return
            
        if len(ids) != len(vectors):
            self.logger.error(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({len(vectors)})")
            return
            
        # Convert to numpy array
        vectors_np = self._convert_vectors(vectors)
        
        # Train index if needed
        self._train_if_needed(vectors_np)
        
        # Get current index size
        current_size = self.index.ntotal
        
        # Update ID map
        for i, doc_id in enumerate(ids):
            self.id_map[current_size + i] = doc_id
            
        # Update metadata map
        if metadatas:
            for doc_id, metadata in zip(ids, metadatas):
                self.metadata_map[doc_id] = metadata
        
        # Add vectors to the index - compatible with faiss-cpu 1.7.4
        self.index.add(vectors_np)
        
        self.logger.info(f"Added {len(ids)} vectors to the index, total: {self.index.ntotal}")
    
    @timer_decorator
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (list): Query embedding vector
            k (int, optional): Number of results to return
            
        Returns:
            list: List of search results with IDs, scores, and metadata
        """
        if not self.index or self.index.ntotal == 0:
            self.logger.warning("Empty index, no results returned")
            return []
            
        try:
            # Convert query vector to numpy array with correct type
            query_np = np.array([query_vector], dtype=np.float32)
            
            # Search the index
            max_results = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_np, max_results)
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices
                if idx < 0:
                    continue
                    
                # Get document ID
                doc_id = self.id_map.get(idx)
                if not doc_id:
                    continue
                    
                # Get metadata
                metadata = self.metadata_map.get(doc_id, {})
                
                # Convert distance to similarity score (FAISS uses L2 distance)
                # Smaller distance means more similar, so we invert it
                max_distance = 100.0  # Arbitrary max distance for normalization
                similarity = max(0.0, 1.0 - (distance / max_distance))
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": similarity,
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching index: {e}")
            return []
    
    @timer_decorator
    def save(self) -> None:
        """Save the index and metadata to disk."""
        if not self.index:
            self.logger.warning("No index to save")
            return
            
        try:
            # Save the FAISS index
            index_path = os.path.join(self.index_dir, f"faiss_{self.index_type}.index")
            faiss.write_index(self.index, index_path)
            
            # Save the ID map
            id_map_path = os.path.join(self.index_dir, "id_map.json")
            with open(id_map_path, 'w') as f:
                # Convert integer keys to strings for JSON
                id_map_str = {str(k): v for k, v in self.id_map.items()}
                json.dump(id_map_str, f)
            
            # Save the metadata map
            metadata_path = os.path.join(self.index_dir, "metadata_map.pickle")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_map, f)
                
            self.logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    @timer_decorator
    def load(self) -> bool:
        """
        Load the index and metadata from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Check if index file exists
            index_path = os.path.join(self.index_dir, f"faiss_{self.index_type}.index")
            if not os.path.exists(index_path):
                self.logger.warning(f"Index file not found at {index_path}")
                return False
                
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load ID map
            id_map_path = os.path.join(self.index_dir, "id_map.json")
            if os.path.exists(id_map_path):
                with open(id_map_path, 'r') as f:
                    # Convert string keys back to integers
                    id_map_str = json.load(f)
                    self.id_map = {int(k): v for k, v in id_map_str.items()}
            
            # Load metadata map
            metadata_path = os.path.join(self.index_dir, "metadata_map.pickle")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata_map = pickle.load(f)
            
            self.logger.info(f"Loaded index with {self.index.ntotal} vectors from {self.index_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            # Reinitialize index
            self._initialize_index()
            return False
    
    @timer_decorator
    def clear(self) -> None:
        """Clear the index and metadata."""
        # Reset the index
        self._initialize_index()
        
        # Clear metadata
        self.id_map = {}
        self.metadata_map = {}
        
        self.logger.info("Cleared index")


class IndexManager:
    """Manager for building and using vector indexes."""
    
    def __init__(self, index: VectorIndex = None):
        """
        Initialize the index manager.
        
        Args:
            index (VectorIndex, optional): Vector index to use
        """
        self.index = index or FAISSIndex()
        self.logger = get_logger("index_manager")
        
        # Try to load existing index
        self.index.load()
    
    @timer_decorator
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks (list): List of document chunks with embeddings
        """
        if not chunks:
            self.logger.warning("No chunks to index")
            return
            
        # Check if chunks have embeddings
        if "embedding" not in chunks[0]:
            self.logger.error("Chunks must have embeddings to be indexed")
            return
            
        # Extract data for indexing
        ids = []
        vectors = []
        metadatas = []
        
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
                
            # Get chunk ID
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
                
            # Get chunk embedding
            embedding = chunk.get("embedding")
            if not embedding:
                continue
                
            # Get chunk metadata (everything except the embedding and text)
            metadata = {k: v for k, v in chunk.items() if k not in ("embedding", "text")}
            
            # Add chunk text to metadata (for retrieval)
            metadata["text"] = chunk.get("text", "")
            
            # Add to lists
            ids.append(chunk_id)
            vectors.append(embedding)
            metadatas.append(metadata)
        
        # Add to index
        if ids and vectors:
            self.index.add(ids, vectors, metadatas)
            
        self.logger.info(f"Indexed {len(ids)} chunks")
    
    @timer_decorator
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding (list): Query embedding vector
            k (int, optional): Number of results to return
            
        Returns:
            list: List of search results
        """
        return self.index.search(query_embedding, k)
    
    def save_index(self) -> None:
        """Save the index to disk."""
        self.index.save()
    
    def clear_index(self) -> None:
        """Clear the index."""
        self.index.clear()








"""
Main application module for the RAG application.
Updated for compatibility with Flask 3.0.0
"""
import os
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

import config
from utils import get_logger
from modules.api import api_bp

# Initialize logger
logger = get_logger("app")

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    # Configure app - updated for Flask 3.0.0
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Add routes - updated for Flask 3.0 compatibility
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/chat')
    def chat():
        """Render the chat page."""
        return render_template('chat.html')
    
    @app.route('/favicon.ico')
    def favicon():
        """Serve favicon."""
        return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                                  'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
    # Add error handlers - updated for Flask 3.0
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('error.html', error_code=404, error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {str(e)}")
        return render_template('error.html', error_code=500, error_message="Server error"), 500
    
    # Custom Jinja filter for date formatting
    @app.template_filter('now')
    def datetime_now(format_string):
        """Return current date formatted."""
        from datetime import datetime
        if format_string == 'year':
            return datetime.now().year
        return datetime.now().strftime(format_string)
    
    # Validate configuration
    if not config.validate_config():
        logger.warning("Application configuration is incomplete. Some features may not work.")
    
    logger.info(f"Application initialized in {config.DEBUG and 'DEBUG' or 'PRODUCTION'} mode")
    return app








"""
Logging utility for the RAG application.
Sets up a standardized logging format and configuration.
Updated for compatibility with loguru 0.7.0
"""
import os
import sys
from datetime import datetime
from loguru import logger

import config

# Remove default logger
logger.remove()

# Set log level from configuration
LOG_LEVEL = getattr(config, "LOG_LEVEL", "INFO")

# Create logs directory if it doesn't exist
logs_dir = os.path.join(config.BASE_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Get current date for log filename
current_date = datetime.now().strftime("%Y-%m-%d")
log_file = os.path.join(logs_dir, f"{current_date}.log")

# Configure logger format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Add console handler
logger.add(
    sys.stderr,
    format=log_format,
    level=LOG_LEVEL,
    colorize=True,
)

# Add file handler - updated for loguru 0.7.0
logger.add(
    log_file,
    format=log_format,
    level=LOG_LEVEL,
    rotation="00:00",  # New file at midnight
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress old log files
    encoding="utf-8"    # Explicit encoding
)

# Function to get logger for a specific module
def get_logger(name):
    """
    Get a configured logger instance for the specified module name.
    
    Args:
        name (str): The name of the module or component
        
    Returns:
        loguru.Logger: A configured logger instance
    """
    return logger.bind(name=name)








"""
Helper functions for the RAG application.
Provides utility functions used across different modules.
Updated for compatibility with BeautifulSoup 4.12.2 and nltk 3.8.1
"""
import re
import time
import json
import hashlib
import unicodedata
from functools import wraps
from datetime import datetime
from pathlib import Path
import os
import uuid

from loguru import logger
import config


def generate_cache_key(text, prefix=""):
    """
    Generate a unique cache key based on text content.
    
    Args:
        text (str): The text to generate a key for
        prefix (str, optional): A prefix to add to the key
        
    Returns:
        str: A unique key suitable for caching
    """
    # Normalize the text to ensure consistent keys
    normalized_text = unicodedata.normalize("NFKD", text.lower().strip())
    
    # Create a hash of the text
    hash_obj = hashlib.md5(normalized_text.encode('utf-8'))
    hash_str = hash_obj.hexdigest()
    
    # Return the key with an optional prefix
    return f"{prefix}_{hash_str}" if prefix else hash_str


def timer_decorator(func):
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func (callable): The function to be timed
        
    Returns:
        callable: The wrapped function with timing functionality
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


def clean_html(html_content):
    """
    Clean HTML content by removing scripts, styles, and unnecessary tags.
    Updated for BeautifulSoup 4.12.2
    
    Args:
        html_content (str): The HTML content to clean
        
    Returns:
        str: Cleaned text content
    """
    from bs4 import BeautifulSoup
    import html2text
    
    if not html_content:
        return ""
    
    # Parse HTML with BeautifulSoup with explicit parser
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Convert to markdown first (preserves some structure)
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_tables = False
    markdown_text = h.handle(str(soup))
    
    # Clean up the markdown text
    return markdown_text.strip()


def extract_code_blocks(text):
    """
    Extract code blocks from markdown text.
    
    Args:
        text (str): Markdown text that may contain code blocks
        
    Returns:
        list: List of dictionaries containing code blocks and their language
    """
    if not text:
        return []
        
    # Pattern to match code blocks with or without language specification
    pattern = r"```(\w*)\n([\s\S]*?)```"
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language.strip() or "text",
            "code": code.strip()
        })
    
    return code_blocks


def extract_tables(html_content):
    """
    Extract tables from HTML content.
    Updated for BeautifulSoup 4.12.2
    
    Args:
        html_content (str): The HTML content with possible tables
        
    Returns:
        list: List of dictionaries containing table data
    """
    from bs4 import BeautifulSoup
    
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    extracted_tables = []
    for table in tables:
        # Extract headers
        headers = []
        header_row = table.find("thead")
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all("th")]
        
        # Extract rows
        rows = []
        body = table.find("tbody")
        if body:
            for row in body.find_all("tr"):
                cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                rows.append(cells)
        else:
            # If no tbody, get all rows directly
            for row in table.find_all("tr"):
                cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                rows.append(cells)
        
        extracted_tables.append({
            "headers": headers,
            "rows": rows
        })
    
    return extracted_tables


def ensure_directory(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        str: The path to the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_to_disk(data, directory, filename=None):
    """
    Save data to disk in JSON format.
    
    Args:
        data (dict or list): The data to save
        directory (str): Directory to save to
        filename (str, optional): Filename to use. If None, generates a UUID
        
    Returns:
        str: Path to the saved file
    """
    # Ensure directory exists
    ensure_directory(directory)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"{uuid.uuid4()}.json"
    
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"
    
    # Create full path
    file_path = os.path.join(directory, filename)
    
    # Save data with error handling
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return None


def load_from_disk(file_path):
    """
    Load JSON data from disk.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict or list: The loaded data, or None if file not found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None


def is_valid_jwt(token):
    """
    Check if a JWT token is valid (not expired).
    Updated for PyJWT 2.8.0
    
    Args:
        token (str): The JWT token to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    import jwt
    from jwt.exceptions import PyJWTError
    
    if not token:
        return False
        
    try:
        # Decode without verification (we just want to check expiry)
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Check if token is expired
        exp = decoded.get("exp", 0)
        current_time = datetime.utcnow().timestamp()
        
        return exp > current_time
    except PyJWTError:
        return False


def format_chunk_for_prompt(chunk, include_metadata=True):
    """
    Format a chunk for inclusion in an LLM prompt.
    
    Args:
        chunk (dict): The chunk to format
        include_metadata (bool): Whether to include metadata
        
    Returns:
        str: Formatted chunk text
    """
    if not chunk:
        return ""
        
    text = chunk.get("text", "")
    
    if include_metadata and "metadata" in chunk:
        metadata = chunk["metadata"]
        source = metadata.get("source", "Unknown")
        title = metadata.get("title", "Untitled")
        url = metadata.get("url", "")
        
        header = f"Source: {source}\nTitle: {title}\n"
        if url:
            header += f"URL: {url}\n"
            
        return f"{header}\n{text}"
    else:
        return text








"""
Lexical search module for the RAG application.
Provides functionality for keyword-based search using BM25 algorithm.
Updated for compatibility with rank-bm25 0.2.2 and nltk 3.8.1
"""
from typing import List, Dict, Any, Set, Optional
import os
import json
import pickle
import time
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

import config
from utils import get_logger, timer_decorator, ensure_directory

# Initialize logger
logger = get_logger("lexical_search")

# Download required NLTK resources with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.error(f"Failed to download NLTK punkt: {e}")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.error(f"Failed to download NLTK stopwords: {e}")

class LexicalSearchRetriever:
    """Retriever that uses BM25 for keyword-based search."""
    
    def __init__(self, index_dir: str = None):
        """
        Initialize the lexical search retriever.
        
        Args:
            index_dir (str, optional): Directory to store the index. Defaults to config value.
        """
        self.index_dir = index_dir or os.path.join(config.CACHE_DIR, "lexical_index")
        ensure_directory(self.index_dir)
        
        self.logger = get_logger("lexical_search_retriever")
        
        # Initialize tokenizer components with error handling for nltk
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.error(f"Error initializing nltk components: {e}")
            # Fallback to empty stopwords if nltk fails
            self.stemmer = PorterStemmer()
            self.stop_words = set()
        
        # Initialize index
        self.bm25_index = None
        self.doc_tokens = []
        self.doc_mapping = {}  # Maps internal indices to document IDs
        self.doc_metadata = {}  # Maps document IDs to metadata
        
        # Load existing index if available
        self.load_index()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for indexing or searching.
        Updated for nltk 3.8.1
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        if not text:
            return []
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers (keep letters and spaces)
            text = re.sub(r'[^a-z\s]', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stop words and stem
            preprocessed_tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 1
            ]
            
            return preprocessed_tokens
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            # Simple fallback if nltk processing fails
            words = text.lower().split()
            return [w for w in words if len(w) > 1]
    
    @timer_decorator
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        Updated for compatibility with rank-bm25 0.2.2
        
        Args:
            chunks (list): List of document chunks
        """
        if not chunks:
            self.logger.warning("No chunks to index")
            return
        
        # Clear existing index if any
        self.clear_index()
        
        # Process each chunk
        doc_tokens = []
        doc_mapping = {}
        doc_metadata = {}
        
        for i, chunk in enumerate(chunks):
            # Get chunk text
            text = chunk.get("text", "")
            if not text:
                continue
            
            # Get chunk ID
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            
            # Preprocess text
            tokens = self._preprocess_text(text)
            if not tokens:
                continue
            
            # Store tokens
            doc_tokens.append(tokens)
            
            # Map internal index to chunk ID
            doc_mapping[i] = chunk_id
            
            # Store metadata
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            metadata["text"] = text  # Include original text for retrieval
            doc_metadata[chunk_id] = metadata
        
        # Build BM25 index with error handling
        try:
            if doc_tokens:
                self.bm25_index = BM25Okapi(doc_tokens)
                self.doc_tokens = doc_tokens
                self.doc_mapping = doc_mapping
                self.doc_metadata = doc_metadata
                
                # Save the index
                self.save_index()
                
                self.logger.info(f"Indexed {len(doc_tokens)} chunks for lexical search")
            else:
                self.logger.warning("No valid documents to index")
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for chunks matching the query.
        Updated for compatibility with rank-bm25 0.2.2
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum BM25 score. Defaults to 0.1.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # Check if index exists
        if not self.bm25_index or not self.doc_tokens:
            self.logger.warning("Lexical index is empty, no results returned")
            return []
        
        # Preprocess query
        start_time = time.time()
        query_tokens = self._preprocess_text(query)
        preprocessing_time = time.time() - start_time
        
        # Handle empty query tokens
        if not query_tokens:
            self.logger.warning("Query contains no meaningful terms after preprocessing")
            return []
        
        # Search using BM25 with error handling
        try:
            start_time = time.time()
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            search_time = time.time() - start_time
            
            # Get top K results with scores above threshold
            results = []
            for i, score in enumerate(bm25_scores):
                if score >= min_score:
                    # Get document ID from internal index
                    doc_id = self.doc_mapping.get(i)
                    if not doc_id:
                        continue
                    
                    # Get metadata
                    metadata = self.doc_metadata.get(doc_id, {})
                    
                    # Normalize score (BM25 scores are not bounded)
                    # This is a simple normalization; adjust as needed
                    normalized_score = min(score / 10.0, 1.0)
                    
                    # Add to results
                    results.append({
                        "id": doc_id,
                        "score": normalized_score,
                        "bm25_score": score,
                        "metadata": metadata
                    })
            
            # Sort by score and limit to k results
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            results = results[:k]
            
            # Log search metrics
            self.logger.debug(f"Lexical search metrics - Preprocessing: {preprocessing_time:.4f}s, Search: {search_time:.4f}s")
            self.logger.info(f"Lexical search found {len(results)} results for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during lexical search: {e}")
            return []
    
    @timer_decorator
    def save_index(self) -> None:
        """
        Save the index to disk.
        """
        try:
            # Check if we have an index to save
            if not self.bm25_index or not self.doc_tokens:
                self.logger.warning("No lexical index to save")
                return
                
            # Save BM25 index
            index_path = os.path.join(self.index_dir, "bm25_index.pickle")
            with open(index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            
            # Save document tokens
            tokens_path = os.path.join(self.index_dir, "doc_tokens.pickle")
            with open(tokens_path, 'wb') as f:
                pickle.dump(self.doc_tokens, f)
            
            # Save document mapping
            mapping_path = os.path.join(self.index_dir, "doc_mapping.json")
            with open(mapping_path, 'w') as f:
                # Convert integer keys to strings for JSON
                mapping_str = {str(k): v for k, v in self.doc_mapping.items()}
                json.dump(mapping_str, f)
            
            # Save document metadata
            metadata_path = os.path.join(self.index_dir, "doc_metadata.pickle")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.doc_metadata, f)
                
            self.logger.info(f"Saved lexical index with {len(self.doc_tokens)} documents")
            
        except Exception as e:
            self.logger.error(f"Error saving lexical index: {e}")
    
    @timer_decorator
    def load_index(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Check if index files exist
            index_path = os.path.join(self.index_dir, "bm25_index.pickle")
            tokens_path = os.path.join(self.index_dir, "doc_tokens.pickle")
            mapping_path = os.path.join(self.index_dir, "doc_mapping.json")
            metadata_path = os.path.join(self.index_dir, "doc_metadata.pickle")
            
            if not all(os.path.exists(p) for p in [index_path, tokens_path, mapping_path, metadata_path]):
                self.logger.warning("One or more lexical index files not found")
                return False
                
            # Load BM25 index
            with open(index_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # Load document tokens
            with open(tokens_path, 'rb') as f:
                self.doc_tokens = pickle.load(f)
            
            # Load document mapping
            with open(mapping_path, 'r') as f:
                # Convert string keys back to integers
                mapping_str = json.load(f)
                self.doc_mapping = {int(k): v for k, v in mapping_str.items()}
            
            # Load document metadata
            with open(metadata_path, 'rb') as f:
                self.doc_metadata = pickle.load(f)
            
            self.logger.info(f"Loaded lexical index with {len(self.doc_tokens)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading lexical index: {e}")
            self.clear_index()
            return False
    
    def clear_index(self) -> None:
        """Clear the in-memory index."""
        self.bm25_index = None
        self.doc_tokens = []
        self.doc_mapping = {}
        self.doc_metadata = {}








"""
Vector search module for the RAG application.
Provides functionality for semantic search using vector embeddings.
Updated for compatibility with sentence-transformers 2.4.1
"""
from typing import List, Dict, Any, Optional
import time

import config
from utils import get_logger, timer_decorator
from modules.processing import EmbeddingProcessor, IndexManager

# Initialize logger
logger = get_logger("vector_search")

class VectorSearchRetriever:
    """Retriever that uses vector similarity search."""
    
    def __init__(self, embedding_processor: EmbeddingProcessor = None, index_manager: IndexManager = None):
        """
        Initialize the vector search retriever.
        
        Args:
            embedding_processor (EmbeddingProcessor, optional): Embedding processor to use
            index_manager (IndexManager, optional): Index manager to use
        """
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        self.index_manager = index_manager or IndexManager()
        self.logger = get_logger("vector_search_retriever")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum similarity score. Defaults to 0.0.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # Get query embedding
        start_time = time.time()
        try:
            query_embedding = self.embedding_processor.embed_query(query)
            embedding_time = time.time() - start_time
            
            # Search the index
            start_time = time.time()
            results = self.index_manager.search(query_embedding, k=k)
            search_time = time.time() - start_time
            
            # Filter by minimum score
            filtered_results = [result for result in results if result.get("score", 0) >= min_score]
            
            # Log search metrics
            self.logger.debug(f"Vector search metrics - Embedding: {embedding_time:.4f}s, Search: {search_time:.4f}s")
            self.logger.info(f"Vector search found {len(filtered_results)} results for query: {query[:50]}...")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            return []
    
    @timer_decorator
    def multi_query_search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search with multiple query variations to improve recall.
        
        Args:
            query (str): Original query text
            k (int, optional): Number of results to return per query variant. Defaults to config value.
            min_score (float, optional): Minimum similarity score. Defaults to 0.0.
            
        Returns:
            list: Merged list of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        try:
            # Generate query variations
            query_variations = self._generate_query_variations(query)
            
            # Get embeddings for all variations
            all_embeddings = self.embedding_processor.embed_queries(query_variations)
            
            # Search with each variation
            all_results = []
            for i, (variation, embedding) in enumerate(zip(query_variations, all_embeddings)):
                # Search the index
                results = self.index_manager.search(embedding, k=k)
                
                # Add query variant info to results
                for result in results:
                    result["query_variant"] = i
                    result["query_text"] = variation
                
                all_results.extend(results)
            
            # Merge results (remove duplicates, keep highest score)
            merged_results = {}
            
            for result in all_results:
                result_id = result.get("id")
                score = result.get("score", 0)
                
                if result_id not in merged_results or score > merged_results[result_id].get("score", 0):
                    merged_results[result_id] = result
            
            # Convert to list and sort by score
            results_list = list(merged_results.values())
            results_list.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Apply minimum score filter
            filtered_results = [result for result in results_list if result.get("score", 0) >= min_score]
            
            # Truncate to k results
            final_results = filtered_results[:k]
            
            self.logger.info(f"Multi-query search found {len(final_results)} results for query: {query[:50]}...")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in multi-query search: {str(e)}")
            return []
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of a query to improve retrieval.
        
        Args:
            query (str): Original query
            
        Returns:
            list: List of query variations
        """
        variations = [query]  # Original query
        
        try:
            # Add a "what is" variant if query is short and doesn't have it
            if len(query) < 100 and not query.lower().startswith("what is"):
                variations.append(f"What is {query}?")
            
            # Add a more detailed variant
            if len(query) < 100:
                variations.append(f"Explain in detail about {query}")
            
            # Add a simpler variant (first 8 words or so)
            words = query.split()
            if len(words) > 8:
                variations.append(" ".join(words[:8]))
            
            # If query is a question, add an instruction variant
            if "?" in query:
                variations.append(query.replace("?", ""))
            
            # Remove duplicates
            unique_variations = list(dict.fromkeys(variations))
            
            self.logger.debug(f"Generated {len(unique_variations)} query variations")
            return unique_variations
            
        except Exception as e:
            self.logger.error(f"Error generating query variations: {str(e)}")
            return [query]  # Return original query on error
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks (list): List of document chunks
        """
        try:
            # Generate embeddings for chunks
            chunks_with_embeddings = self.embedding_processor.process_chunks(chunks)
            
            # Index the chunks
            self.index_manager.index_chunks(chunks_with_embeddings)
            
            # Save the index
            self.index_manager.save_index()
            
        except Exception as e:
            self.logger.error(f"Error indexing chunks: {str(e)}")











"""
Hybrid search module for the RAG application.
Combines vector and lexical search for improved retrieval.
Updated for compatibility with numpy 1.24.4
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import time

import config
from utils import get_logger, timer_decorator
from modules.retrieval.vector_search import VectorSearchRetriever
from modules.retrieval.lexical_search import LexicalSearchRetriever

# Initialize logger
logger = get_logger("hybrid_search")

class HybridSearchRetriever:
    """Retriever that combines vector and lexical search."""
    
    def __init__(
        self,
        vector_retriever: VectorSearchRetriever = None,
        lexical_retriever: LexicalSearchRetriever = None,
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3
    ):
        """
        Initialize the hybrid search retriever.
        
        Args:
            vector_retriever (VectorSearchRetriever, optional): Vector search retriever
            lexical_retriever (LexicalSearchRetriever, optional): Lexical search retriever
            vector_weight (float, optional): Weight for vector search scores. Defaults to 0.7.
            lexical_weight (float, optional): Weight for lexical search scores. Defaults to 0.3.
        """
        self.vector_retriever = vector_retriever or VectorSearchRetriever()
        self.lexical_retriever = lexical_retriever or LexicalSearchRetriever()
        
        # Ensure weights sum to 1.0
        total_weight = vector_weight + lexical_weight
        self.vector_weight = vector_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
        
        self.logger = get_logger("hybrid_search_retriever")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for chunks using both vector and lexical search.
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum combined score. Defaults to 0.0.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        try:
            # We'll retrieve more results from each search to get better candidates
            retrieval_k = max(k * 2, 10)
            
            # Start vector search
            start_time = time.time()
            vector_results = self.vector_retriever.search(query, k=retrieval_k)
            vector_time = time.time() - start_time
            
            # Start lexical search
            start_time = time.time()
            lexical_results = self.lexical_retriever.search(query, k=retrieval_k)
            lexical_time = time.time() - start_time
            
            # Merge results with weighted scores
            start_time = time.time()
            merged_results = self._merge_results(vector_results, lexical_results)
            merge_time = time.time() - start_time
            
            # Filter by minimum score
            filtered_results = [result for result in merged_results if result.get("score", 0) >= min_score]
            
            # Sort by score and limit to k results
            filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = filtered_results[:k]
            
            # Log search metrics
            self.logger.debug(
                f"Hybrid search metrics - Vector: {vector_time:.4f}s, "
                f"Lexical: {lexical_time:.4f}s, Merge: {merge_time:.4f}s"
            )
            self.logger.info(
                f"Hybrid search found {len(final_results)} results "
                f"(from {len(vector_results)} vector, {len(lexical_results)} lexical) "
                f"for query: {query[:50]}..."
            )
            
            return final_results
        
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to vector search only if hybrid fails
            return self.vector_retriever.search(query, k=k, min_score=min_score)
    
    @timer_decorator
    def multi_query_search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search with multiple query variations using both vector and lexical search.
        
        Args:
            query (str): Original query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum combined score. Defaults to 0.0.
            
        Returns:
            list: Merged list of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        try:
            # We'll retrieve more results to get better candidates
            retrieval_k = max(k * 2, 10)
            
            # Vector search with multiple queries
            start_time = time.time()
            vector_results = self.vector_retriever.multi_query_search(query, k=retrieval_k)
            vector_time = time.time() - start_time
            
            # Regular lexical search (already handles variations to some extent)
            start_time = time.time()
            lexical_results = self.lexical_retriever.search(query, k=retrieval_k)
            lexical_time = time.time() - start_time
            
            # Merge results with weighted scores
            start_time = time.time()
            merged_results = self._merge_results(vector_results, lexical_results)
            merge_time = time.time() - start_time
            
            # Filter by minimum score
            filtered_results = [result for result in merged_results if result.get("score", 0) >= min_score]
            
            # Sort by score and limit to k results
            filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = filtered_results[:k]
            
            # Log search metrics
            self.logger.debug(
                f"Hybrid multi-query search metrics - Vector: {vector_time:.4f}s, "
                f"Lexical: {lexical_time:.4f}s, Merge: {merge_time:.4f}s"
            )
            self.logger.info(
                f"Hybrid multi-query search found {len(final_results)} results "
                f"for query: {query[:50]}..."
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid multi-query search: {str(e)}")
            # Fallback to vector multi-query search only if hybrid fails
            return self.vector_retriever.multi_query_search(query, k=k, min_score=min_score)
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge results from vector and lexical search with weighted scoring.
        
        Args:
            vector_results (list): Results from vector search
            lexical_results (list): Results from lexical search
            
        Returns:
            list: Merged results with combined scores
        """
        # Create a map of document IDs to scores
        result_map = {}
        
        # Process vector results
        for result in vector_results:
            doc_id = result.get("id")
            if not doc_id:
                continue
                
            score = result.get("score", 0)
            metadata = result.get("metadata", {})
            
            result_map[doc_id] = {
                "id": doc_id,
                "vector_score": score,
                "lexical_score": 0,
                "metadata": metadata,
                "sources": ["vector"]
            }
        
        # Process lexical results
        for result in lexical_results:
            doc_id = result.get("id")
            if not doc_id:
                continue
                
            score = result.get("score", 0)
            metadata = result.get("metadata", {})
            
            if doc_id in result_map:
                # Update existing entry
                result_map[doc_id]["lexical_score"] = score
                result_map[doc_id]["sources"].append("lexical")
                
                # Use metadata from lexical result if vector doesn't have it
                if not result_map[doc_id]["metadata"] and metadata:
                    result_map[doc_id]["metadata"] = metadata
            else:
                # Create new entry
                result_map[doc_id] = {
                    "id": doc_id,
                    "vector_score": 0,
                    "lexical_score": score,
                    "metadata": metadata,
                    "sources": ["lexical"]
                }
        
        # Calculate combined scores
        results = []
        for doc_id, data in result_map.items():
            vector_score = data.get("vector_score", 0)
            lexical_score = data.get("lexical_score", 0)
            
            # Combine scores
            combined_score = (vector_score * self.vector_weight) + (lexical_score * self.lexical_weight)
            
            # Create final result item
            result = {
                "id": doc_id,
                "score": combined_score,
                "vector_score": vector_score,
                "lexical_score": lexical_score,
                "metadata": data.get("metadata", {}),
                "sources": data.get("sources", [])
            }
            
            results.append(result)
        
        return results
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks for both vector and lexical search.
        
        Args:
            chunks (list): List of document chunks
        """
        try:
            # First, index for vector search
            self.vector_retriever.index_chunks(chunks)
            
            # Then, index for lexical search
            self.lexical_retriever.index_chunks(chunks)
            
            self.logger.info(f"Indexed {len(chunks)} chunks for hybrid search")
            
        except Exception as e:
            self.logger.error(f"Error indexing chunks for hybrid search: {str(e)}")


class ReRanker:
    """Re-rank search results for improved relevance."""
    
    def __init__(self, vector_retriever: VectorSearchRetriever = None):
        """
        Initialize the re-ranker.
        
        Args:
            vector_retriever (VectorSearchRetriever, optional): Vector search retriever for embeddings
        """
        self.vector_retriever = vector_retriever or VectorSearchRetriever()
        self.logger = get_logger("reranker")
    
    @timer_decorator
    def rerank(self, query: str, results: List[Dict[str, Any]], k: int = None) -> List[Dict[str, Any]]:
        """
        Re-rank search results for improved relevance.
        Updated for numpy 1.24.4 compatibility
        
        Args:
            query (str): Query text
            results (list): Initial search results
            k (int, optional): Number of results to return. Defaults to length of results.
            
        Returns:
            list: Re-ranked results
        """
        if not results:
            return []
            
        if k is None:
            k = len(results)
            
        try:
            # Extract texts from results
            texts = []
            for result in results:
                # Get text from metadata
                metadata = result.get("metadata", {})
                text = metadata.get("text", "")
                
                # If no text in metadata, try to get it from other fields
                if not text and "text" in result:
                    text = result.get("text", "")
                
                texts.append(text)
            
            # Generate query and document embeddings
            query_embedding = self.vector_retriever.embedding_processor.embed_query(query)
            doc_embeddings = self.vector_retriever.embedding_processor.embed_queries(texts)
            
            # Calculate more precise similarity scores
            new_scores = []
            for i, doc_embedding in enumerate(doc_embeddings):
                # Compute cosine similarity (dot product of normalized vectors)
                similarity = self._compute_similarity(query_embedding, doc_embedding)
                new_scores.append(similarity)
            
            # Add new scores to results
            for i, (result, new_score) in enumerate(zip(results, new_scores)):
                result["rerank_score"] = new_score
                
                # Optionally adjust the original score
                # This gives more weight to the re-ranking score
                result["original_score"] = result.get("score", 0)
                result["score"] = (result.get("score", 0) * 0.4) + (new_score * 0.6)
            
            # Sort by new scores
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Return top k results
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            return results[:k]  # Return original results on error
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        Updated for numpy 1.24.4
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Cosine similarity (between -1 and 1)
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays with explicit dtype
            v1 = np.array(vec1, dtype=np.float64)
            v2 = np.array(vec2, dtype=np.float64)
            
            # Compute dot product
            dot_product = np.dot(v1, v2)
            
            # Compute magnitudes
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if mag1 * mag2 == 0:
                return 0
            
            # Compute cosine similarity
            cos_sim = dot_product / (mag1 * mag2)
            
            # Ensure result is in valid range
            return float(max(-1.0, min(1.0, cos_sim)))
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0  # Return zero similarity on error











"""
API schemas for the RAG application.
Updated for compatibility with pydantic 1.10.8
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Schema for query request."""
    
    query: str = Field(..., description="The query text")
    sources: List[str] = Field(
        default=["confluence", "remedy"],
        description="Sources to search (confluence, remedy)"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to retrieve",
        ge=1,
        le=20
    )
    mode: str = Field(
        default="hybrid",
        description="Search mode (vector, lexical, hybrid)"
    )
    multi_query: bool = Field(
        default=False,
        description="Whether to use multi-query retrieval"
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Prompt template to use (general, technical, concise, customer_support)"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for generation",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate",
        ge=1,
        le=8192
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    # Validator for mode field
    @validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ["vector", "lexical", "hybrid"]
        if v not in allowed_modes:
            raise ValueError(f"Mode must be one of {allowed_modes}")
        return v
    
    # Validator for prompt_template field
    @validator('prompt_template')
    def validate_prompt_template(cls, v):
        if v is not None:
            allowed_templates = ["general", "technical", "concise", "customer_support"]
            if v not in allowed_templates:
                raise ValueError(f"Prompt template must be one of {allowed_templates}")
        return v


class QueryResponse(BaseModel):
    """Schema for query response."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used for the answer"
    )
    query: str = Field(..., description="Original query")
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information"
    )
    
    class Config:
        """Pydantic config"""
        json_encoders = {
            # Handle types that might not be JSON serializable
            set: list,
            # Add any other custom encoders if needed
        }


class ChatMessage(BaseModel):
    """Schema for chat message."""
    
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    
    # Validator for role field
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ["user", "assistant", "system"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


class ChatRequest(BaseModel):
    """Schema for chat request."""
    
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    sources: List[str] = Field(
        default=["confluence", "remedy"],
        description="Sources to search (confluence, remedy)"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to retrieve",
        ge=1,
        le=20
    )
    mode: str = Field(
        default="hybrid",
        description="Search mode (vector, lexical, hybrid)"
    )
    multi_query: bool = Field(
        default=False,
        description="Whether to use multi-query retrieval"
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Prompt template to use (general, technical, concise, customer_support)"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for generation",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate",
        ge=1,
        le=8192
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    # Validator for mode field
    @validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ["vector", "lexical", "hybrid"]
        if v not in allowed_modes:
            raise ValueError(f"Mode must be one of {allowed_modes}")
        return v
    
    # Validator for prompt_template field
    @validator('prompt_template')
    def validate_prompt_template(cls, v):
        if v is not None:
            allowed_templates = ["general", "technical", "concise", "customer_support"]
            if v not in allowed_templates:
                raise ValueError(f"Prompt template must be one of {allowed_templates}")
        return v


class ChatResponse(BaseModel):
    """Schema for chat response."""
    
    message: ChatMessage = Field(..., description="Assistant message")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used for the answer"
    )
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information"
    )
    
    class Config:
        """Pydantic config"""
        json_encoders = {
            # Handle types that might not be JSON serializable
            set: list,
            # Add any other custom encoders if needed
        }


class IndexRequest(BaseModel):
    """Schema for index request."""
    
    sources: List[str] = Field(
        default=["confluence", "remedy"],
        description="Sources to index (confluence, remedy)"
    )
    confluence_space_key: Optional[str] = Field(
        default=None,
        description="Confluence space key to index"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of documents to index"
    )
    chunking_strategy: str = Field(
        default="semantic",
        description="Chunking strategy (simple, sentence, semantic, hierarchical)"
    )
    force_reindex: bool = Field(
        default=False,
        description="Whether to force reindexing"
    )
    
    # Validator for chunking_strategy field
    @validator('chunking_strategy')
    def validate_chunking_strategy(cls, v):
        allowed_strategies = ["simple", "sentence", "semantic", "hierarchical"]
        if v not in allowed_strategies:
            raise ValueError(f"Chunking strategy must be one of {allowed_strategies}")
        return v


class IndexResponse(BaseModel):
    """Schema for index response."""
    
    status: str = Field(..., description="Status of the indexing operation")
    message: str = Field(..., description="Status message")
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Document counts by source"
    )
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information"
    )


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Status of the application")
    version: str = Field(..., description="Application version")
    endpoints: List[str] = Field(
        default_factory=list,
        description="Available API endpoints"
    )
    sources: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of data sources"
    )
    index_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Index statistics"
    )











"""
Run script for the RAG application.
Updated for compatibility with Flask 3.0.0
"""
import os
from app import create_app
import config

# Import logger after config to ensure configuration is loaded
from utils import get_logger

# Initialize logger
logger = get_logger("run")

# Only run the app if this file is executed directly
if __name__ == '__main__':
    # Create the Flask application
    app = create_app()
    
    # Default port from config or environment
    port = int(os.environ.get("PORT", config.PORT))
    
    # Run the application with explicit host and port
    logger.info(f"Starting application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=config.DEBUG)