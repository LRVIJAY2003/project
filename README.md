"""
Google Gemini integration for the RAG application.
Updated for compatibility with google-cloud-aiplatform==1.71.1
"""
from typing import List, Dict, Any, Optional, Union, Generator
import os
import time

from google.api_core.exceptions import GoogleAPIError
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
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
        
        # Initialize Vertex AI
        try:
            aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
            self.logger.info(f"Initialized Vertex AI for project {config.PROJECT_ID} in {config.REGION}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
        
        # Initialize prediction client
        try:
            self.prediction_client = aiplatform.gapic.PredictionServiceClient()
            self.endpoint = f"projects/{config.PROJECT_ID}/locations/{config.REGION}/publishers/google/models/{self.model_name}"
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
            # Create predict request
            instance = predict.instance.TextPredictionInstance(
                content=prompt,
            ).to_value()
            
            parameters = predict.params.TextGenerationParams(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.9,
                top_k=40
            ).to_value()
            
            # Create request
            request = {
                "endpoint": self.endpoint,
                "instances": [instance],
                "parameters": parameters
            }
            
            # Generate response
            if stream:
                # For streaming, we'll use a custom approach
                def generate_stream():
                    response = self.prediction_client.predict(request=request)
                    for chunk in response:
                        yield chunk.predictions[0]
                
                return generate_stream()
            else:
                response = self.prediction_client.predict(request=request)
                return response.predictions[0]
                
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
            # Create a formatted prompt from the messages
            formatted_prompt = ""
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    formatted_prompt += f"User: {content}\n\n"
                elif role in ["assistant", "model"]:
                    formatted_prompt += f"Assistant: {content}\n\n"
                elif role == "system":
                    formatted_prompt += f"System: {content}\n\n"
            
            # Add final assistant prompt
            formatted_prompt += "Assistant: "
            
            # Generate response using the text generation endpoint
            return self.generate_text(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
                
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
        
        # Create system message with context and instructions
        system_with_context = f"""
        {system_prompt}
        
        Here are the reference documents:
        
        {context_str}
        
        Please provide answers based on the provided documents.
        If the documents don't provide enough information, say so clearly.
        Include citations to the relevant document numbers in your response.
        """.strip()
        
        # Create a new messages list with enhanced first message
        new_messages = []
        
        # Add the chat messages, but prepend the context to the first user message
        first_user_msg_found = False
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user" and not first_user_msg_found:
                first_user_msg_found = True
                new_content = f"With this context:\n\n{system_with_context}\n\nMy question is: {content}"
                new_messages.append({"role": role, "content": new_content})
            else:
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
Embedding module for the RAG application.
Updated for compatibility with transformers==4.14.1
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
        
        # Initialize the model
        try:
            # Use local cache_folder to avoid huggingface hub issues
            model_cache_dir = os.path.join(config.CACHE_DIR, "models")
            ensure_directory(model_cache_dir)
            
            self.model = SentenceTransformer(self.model_name, cache_folder=model_cache_dir)
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
            # Use encode with convert_to_numpy=True and normalize_embeddings=True
            # These are compatible with older transformers versions
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
            
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
                # Modified for compatibility
                new_embeddings = self.model.encode(
                    to_embed,
                    batch_size=16,  # Smaller batch size for compatibility
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
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
Configuration module for the RAG application.
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

# Flask configuration
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

# Embedding model settings
# Using smaller model compatible with older transformers version
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for the embedding model chosen

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
NUM_RESULTS = 5
TEMPERATURE = 0.2

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
        from loguru import logger
        missing_vars_str = ", ".join(missing_vars)
        logger.warning(f"Missing required environment variables: {missing_vars_str}")
        
    return not missing_vars








"""
Processing package for the RAG application.
Updated for compatibility with installed package versions.
"""
from .chunking import (
    ChunkingStrategy,
    SimpleChunker,
    SentenceChunker,
    SemanticChunker,
    HierarchicalChunker,
    ChunkerFactory
)
from .embedding import (
    EmbeddingModel,
    SentenceTransformerEmbedding,
    EmbeddingProcessor
)
from .indexing import (
    VectorIndex,
    FAISSIndex,
    IndexManager
)

__all__ = [
    'ChunkingStrategy',
    'SimpleChunker',
    'SentenceChunker',
    'SemanticChunker',
    'HierarchicalChunker',
    'ChunkerFactory',
    'EmbeddingModel',
    'SentenceTransformerEmbedding',
    'EmbeddingProcessor',
    'VectorIndex',
    'FAISSIndex',
    'IndexManager'
]







"""
Run script for the RAG application.
Updated for compatibility with installed package versions.
"""
import os
import sys
from app import create_app
import config
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=config.LOG_LEVEL)  # Add stderr handler
logger.add(os.path.join("logs", "{time:YYYY-MM-DD}.log"), rotation="00:00", level=config.LOG_LEVEL)  # Add file handler

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting RAG application")
    
    # Initialize application
    app = create_app()
    
    # Run application
    logger.info(f"Starting application on port {config.PORT}")
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)



