"""
Embedding module for the RAG application.
Completely rewritten for compatibility with minimal dependencies.
"""
from typing import List, Dict, Any, Union, Optional
import os
import json
import numpy as np
import time
import hashlib
from collections import Counter

import config
from utils import get_logger, timer_decorator, generate_cache_key, ensure_directory

# Initialize logger
logger = get_logger("embedding")

class FallbackEmbedding:
    """
    Fallback embedding class that doesn't rely on external ML libraries.
    Uses a simple approach to create vector representations of text.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the fallback embedding model.
        
        Args:
            dimension (int): Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.logger = get_logger("fallback_embedding")
        self.logger.warning("Using fallback embedding - this will reduce search quality")
        
        # Cache directory
        self.cache_dir = os.path.join(config.CACHE_DIR, "embeddings")
        ensure_directory(self.cache_dir)
        
        # Create a simple vocabulary for hashing words to dimensions
        self._create_vocabulary()
    
    def _create_vocabulary(self):
        """Create a deterministic mapping from words to dimensions."""
        self.word_to_dim = {}
        
        # These are common English words that will be mapped to specific dimensions
        # Using common words ensures more consistent embeddings
        common_words = [
            "the", "of", "and", "a", "to", "in", "is", "that", "it", "was", "for",
            "on", "are", "as", "with", "they", "be", "at", "this", "from", "have",
            "or", "by", "one", "had", "not", "but", "what", "all", "were", "when",
            "we", "there", "can", "an", "your", "which", "their", "said", "if", "do",
            "will", "each", "about", "how", "up", "out", "them", "then", "she", "many",
            "some", "so", "would", "other", "into", "has", "more", "two", "time", "like",
            "no", "just", "him", "know", "take", "people", "year", "get", "through", "also",
            "back", "after", "use", "our", "work", "first", "well", "way", "even", "new",
            "want", "any", "these", "give", "day", "most", "us", "very", "important", "information",
            "help", "question", "system", "better", "never", "try", "problem", "need", "right", "too",
            "could", "should", "call", "different", "general", "specific", "example", "case", "study", "research",
            "data", "report", "result", "analysis", "development", "management", "company", "business", "service", "process",
            "issue", "market", "customer", "product", "project", "solution", "technology", "function", "application", "feature"
        ]
        
        # Create a deterministic mapping
        for i, word in enumerate(common_words):
            # Use modulo to handle the case where we have more words than dimensions
            self.word_to_dim[word] = i % self.dimension
    
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
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of word tokens
        """
        # Simple tokenization - convert to lowercase and split on spaces and punctuation
        text = text.lower()
        
        # Replace common punctuation with spaces
        for char in ".,;:!?()[]{}-\"'":
            text = text.replace(char, " ")
        
        # Split on whitespace and filter out empty tokens
        tokens = [token.strip() for token in text.split()]
        return [token for token in tokens if token]
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        # Initialize a zero vector
        embedding = np.zeros(self.dimension)
        
        # Tokenize the text
        tokens = self._tokenize(text)
        
        if not tokens:
            return embedding.tolist()
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # For each token, add its contribution to the embedding
        for token, count in token_counts.items():
            # Get the dimension for this token, or hash it if not in vocabulary
            if token in self.word_to_dim:
                dim = self.word_to_dim[token]
            else:
                # Hash the token to get a consistent dimension
                hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
                dim = hash_val % self.dimension
            
            # Add the count to that dimension
            embedding[dim] += count
        
        # Normalize for document length
        if np.sum(embedding) > 0:
            embedding = embedding / np.sqrt(np.sum(np.square(embedding)))
        
        # Add some special features
        if len(text) > 0:
            # Feature for text length - longer texts often more informative
            embedding[0] = min(1.0, len(text) / 1000.0)
            
            # Feature for question detection
            if '?' in text:
                embedding[1] = 0.8
            
            # Feature for keyword density
            if len(tokens) > 0:
                unique_ratio = len(set(tokens)) / len(tokens)
                embedding[2] = unique_ratio
        
        # Final normalization to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def encode(self, texts, **kwargs):
        """
        Generate embeddings for texts (same interface as SentenceTransformer).
        
        Args:
            texts: Text or list of texts to embed
            **kwargs: Ignored kwargs for compatibility
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            return np.array(self._create_embedding(texts))
        
        # Handle list of texts
        embeddings = [self._create_embedding(text) for text in texts]
        return np.array(embeddings)
    
    def get_sentence_embedding_dimension(self):
        """Get embedding dimension (same interface as SentenceTransformer)."""
        return self.dimension


class EmbeddingProcessor:
    """Process documents and generate embeddings for chunks."""
    
    def __init__(self):
        """Initialize the embedding processor."""
        self.logger = get_logger("embedding_processor")
        self.model = self._initialize_model()
        self.cache_dir = os.path.join(config.CACHE_DIR, "embeddings")
        ensure_directory(self.cache_dir)
    
    def _initialize_model(self):
        """
        Try to initialize a proper embedding model, falling back to simple model if needed.
        
        Returns:
            object: An embedding model with encode() method
        """
        # First try to import the actual sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.logger.info(f"Successfully loaded SentenceTransformer model: {config.EMBEDDING_MODEL}")
            return model
        except Exception as primary_error:
            self.logger.error(f"Could not load SentenceTransformer: {primary_error}")
            
            # Try a simpler transformer model
            try:
                from sentence_transformers import SentenceTransformer
                simpler_model = "paraphrase-MiniLM-L3-v2"
                model = SentenceTransformer(simpler_model)
                self.logger.info(f"Successfully loaded fallback SentenceTransformer model: {simpler_model}")
                return model
            except Exception as secondary_error:
                self.logger.error(f"Could not load fallback SentenceTransformer: {secondary_error}")
                
                # Fall back to our basic embedder
                self.logger.warning("Using basic fallback embedding model - search quality will be reduced")
                return FallbackEmbedding(dimension=config.EMBEDDING_DIMENSION)
    
    def _get_cache_path(self, text: str) -> str:
        """Get cache path for an embedding."""
        key = generate_cache_key(text, prefix="emb")
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load embedding from cache."""
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
        """Save embedding to cache."""
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
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query (str): Query text
            
        Returns:
            list: Query embedding vector
        """
        if not query or not query.strip():
            # Return zero vector for empty text
            return [0.0] * config.EMBEDDING_DIMENSION
        
        # Check cache first
        cached_embedding = self._load_from_cache(query)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Use the model to generate embedding
            embedding = self.model.encode(query).tolist()
            
            # Save to cache
            self._save_to_cache(query, embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            # Create a simple fallback embedding
            fallback = FallbackEmbedding(dimension=config.EMBEDDING_DIMENSION)
            embedding = fallback._create_embedding(query)
            self._save_to_cache(query, embedding)
            return embedding
    
    @timer_decorator
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.
        
        Args:
            queries (list): List of query texts
            
        Returns:
            list: List of query embedding vectors
        """
        return [self.embed_query(query) for query in queries]
    
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
        
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # Get text from chunk
                text = chunk.get("text", "")
                if not text:
                    continue
                
                # Generate embedding
                embedding = self.embed_query(text)
                
                # Add embedding to chunk
                chunk_with_embedding = dict(chunk)
                chunk_with_embedding["embedding"] = embedding
                
                processed_chunks.append(chunk_with_embedding)
                
            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}")
        
        self.logger.info(f"Generated embeddings for {len(processed_chunks)} of {len(chunks)} chunks")
        return processed_chunks











"""
Main application module for the RAG application.
Updated with robust error handling to prevent crashes.
"""
import os
import sys
import traceback
from flask import Flask, render_template, send_from_directory, jsonify

import config
from utils import get_logger

# Initialize logger
logger = get_logger("app")

def create_app():
    """Create and configure the Flask application with robust error handling."""
    try:
        app = Flask(__name__, static_folder='static', template_folder='templates')
        
        # Configure app
        app.config['SECRET_KEY'] = config.SECRET_KEY
        app.config['DEBUG'] = config.DEBUG
        
        # Enable CORS if available
        try:
            from flask_cors import CORS
            CORS(app)
            logger.info("CORS initialized")
        except ImportError:
            logger.warning("CORS not available, continuing without it")
        
        # Register blueprints with error handling
        try:
            from modules.api import api_bp
            app.register_blueprint(api_bp)
            logger.info("API blueprint registered")
        except Exception as e:
            logger.error(f"Error registering API blueprint: {e}")
            traceback.print_exc()
            
            # Create a fallback API blueprint if the main one fails
            from flask import Blueprint
            fallback_bp = Blueprint('fallback_api', __name__, url_prefix='/api')
            
            @fallback_bp.route('/health', methods=['GET'])
            def fallback_health():
                return jsonify({
                    "status": "degraded",
                    "message": "System running in degraded mode due to initialization errors",
                    "version": "1.0.0"
                })
                
            app.register_blueprint(fallback_bp)
            logger.info("Fallback API blueprint registered")
        
        # Add routes - with try/except for each route
        @app.route('/')
        def index():
            """Render the main page."""
            try:
                return render_template('index.html')
            except Exception as e:
                logger.error(f"Error rendering index page: {e}")
                return f"<html><body><h1>Error</h1><p>Could not load index page: {str(e)}</p></body></html>"
        
        @app.route('/chat')
        def chat():
            """Render the chat page."""
            try:
                return render_template('chat.html')
            except Exception as e:
                logger.error(f"Error rendering chat page: {e}")
                return f"<html><body><h1>Error</h1><p>Could not load chat page: {str(e)}</p></body></html>"
        
        @app.route('/favicon.ico')
        def favicon():
            """Serve favicon."""
            try:
                return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                                          'favicon.ico', mimetype='image/vnd.microsoft.icon')
            except Exception as e:
                logger.error(f"Error serving favicon: {e}")
                # Return a 1x1 transparent pixel as fallback
                return app.response_class(
                    response=b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x21\xF9\x04\x01\x00\x00\x00\x00\x2C\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3B',
                    status=200,
                    mimetype='image/gif'
                )
        
        # Add error handlers
        @app.errorhandler(404)
        def page_not_found(e):
            """Handle 404 errors."""
            try:
                return render_template('error.html', error_code=404, error_message="Page not found"), 404
            except Exception as render_error:
                logger.error(f"Error rendering 404 page: {render_error}")
                return "404 - Page Not Found", 404
        
        @app.errorhandler(500)
        def server_error(e):
            """Handle 500 errors."""
            logger.error(f"Server error: {str(e)}")
            try:
                return render_template('error.html', error_code=500, error_message="Server error"), 500
            except Exception as render_error:
                logger.error(f"Error rendering 500 page: {render_error}")
                return "500 - Server Error", 500
        
        # Catch-all error handler for all exceptions
        @app.errorhandler(Exception)
        def handle_exception(e):
            """Handle all uncaught exceptions."""
            logger.error(f"Unhandled exception: {str(e)}")
            traceback.print_exc()
            try:
                return render_template('error.html', error_code=500, error_message=f"Server error: {str(e)}"), 500
            except Exception:
                return f"500 - Server Error: {str(e)}", 500
        
        # Custom Jinja filter for date formatting
        @app.template_filter('now')
        def datetime_now(format_string):
            """Return current date formatted."""
            try:
                from datetime import datetime
                if format_string == 'year':
                    return datetime.now().year
                return datetime.now().strftime(format_string)
            except Exception as e:
                logger.error(f"Error in datetime filter: {e}")
                return "YYYY"
        
        # Validate configuration
        try:
            if not config.validate_config():
                logger.warning("Application configuration is incomplete. Some features may not work.")
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
        
        logger.info(f"Application initialized in {config.DEBUG and 'DEBUG' or 'PRODUCTION'} mode")
        return app
        
    except Exception as e:
        logger.critical(f"Critical error initializing application: {e}")
        traceback.print_exc()
        
        # Create a minimal emergency application
        emergency_app = Flask(__name__)
        
        @emergency_app.route('/', defaults={'path': ''})
        @emergency_app.route('/<path:path>')
        def fallback(path):
            return """
            <html>
                <head><title>System Error</title></head>
                <body>
                    <h1>System Error</h1>
                    <p>The application could not start due to initialization errors.</p>
                    <p>Please check the logs and try again.</p>
                </body>
            </html>
            """, 500
            
        return emergency_app








"""
Run script for the RAG application.
Updated with comprehensive error handling.
"""
import os
import sys
import traceback

# Set up basic logging before importing other modules
# This ensures we can log errors even if other imports fail
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
basic_logger = logging.getLogger("run_bootstrap")

try:
    # Import config first to ensure environment variables are loaded
    import config
    
    # Then try to import the custom logger
    try:
        from utils import get_logger
        logger = get_logger("run")
    except ImportError:
        # Fall back to basic logger if custom logger fails
        logger = basic_logger
        logger.warning("Custom logger not available, using basic logger")
    
    # Import the app factory function
    try:
        from app import create_app
    except ImportError as e:
        logger.critical(f"Failed to import app module: {e}")
        sys.exit(1)
        
except Exception as e:
    basic_logger.critical(f"Fatal error during initialization: {e}")
    traceback.print_exc()
    sys.exit(1)

def get_port():
    """Get port from environment or config with error handling."""
    try:
        return int(os.environ.get("PORT", config.PORT))
    except (ValueError, AttributeError):
        # Default to 5000 if port is not valid
        return 5000

# Only run the app if this file is executed directly
if __name__ == '__main__':
    try:
        # Create the Flask application
        app = create_app()
        
        # Get port
        port = get_port()
        
        # Run the application with explicit host and port
        logger.info(f"Starting application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=config.DEBUG)
        
    except Exception as e:
        logger.critical(f"Fatal error starting application: {e}")
        traceback.print_exc()
        sys.exit(1)








"""
API routes for the RAG application.
Updated with comprehensive error handling to prevent crashes.
"""
from typing import List, Dict, Any, Optional, Union, Generator
import time
import json
import traceback
from flask import Blueprint, request, jsonify, Response, stream_with_context, current_app

import config
from utils import get_logger, timer_decorator

# Initialize logger
logger = get_logger("api_routes")

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize components with robust error handling
try:
    from modules.data_sources import ConfluenceClient, ConfluenceContentProcessor
    confluence_client = ConfluenceClient()
    confluence_processor = ConfluenceContentProcessor(confluence_client)
    logger.info("Confluence components initialized")
except Exception as e:
    logger.error(f"Error initializing Confluence components: {e}")
    confluence_client = None
    confluence_processor = None

try:
    from modules.data_sources import RemedyClient, RemedyContentProcessor
    remedy_client = RemedyClient()
    remedy_processor = RemedyContentProcessor(remedy_client)
    logger.info("Remedy components initialized")
except Exception as e:
    logger.error(f"Error initializing Remedy components: {e}")
    remedy_client = None
    remedy_processor = None

try:
    from modules.processing import EmbeddingProcessor
    embedding_processor = EmbeddingProcessor()
    logger.info("Embedding processor initialized")
except Exception as e:
    logger.error(f"Error initializing Embedding processor: {e}")
    embedding_processor = None

try:
    from modules.processing import IndexManager
    index_manager = IndexManager()
    logger.info("Index manager initialized")
except Exception as e:
    logger.error(f"Error initializing Index manager: {e}")
    index_manager = None

try:
    from modules.retrieval import VectorSearchRetriever
    vector_retriever = VectorSearchRetriever(embedding_processor, index_manager)
    logger.info("Vector retriever initialized")
except Exception as e:
    logger.error(f"Error initializing Vector retriever: {e}")
    vector_retriever = None

try:
    from modules.retrieval import LexicalSearchRetriever
    lexical_retriever = LexicalSearchRetriever()
    logger.info("Lexical retriever initialized")
except Exception as e:
    logger.error(f"Error initializing Lexical retriever: {e}")
    lexical_retriever = None

try:
    from modules.retrieval import HybridSearchRetriever, ReRanker
    hybrid_retriever = None
    if vector_retriever and lexical_retriever:
        hybrid_retriever = HybridSearchRetriever(vector_retriever, lexical_retriever)
        logger.info("Hybrid retriever initialized")
    reranker = ReRanker(vector_retriever) if vector_retriever else None
    logger.info("Reranker initialized")
except Exception as e:
    logger.error(f"Error initializing Hybrid retriever or Reranker: {e}")
    hybrid_retriever = None
    reranker = None

try:
    from modules.llm import GeminiClient, RAGGenerator, SYSTEM_PROMPTS
    gemini_client = GeminiClient()
    rag_generator = RAGGenerator(gemini_client)
    logger.info("LLM components initialized")
except Exception as e:
    logger.error(f"Error initializing LLM components: {e}")
    gemini_client = None
    rag_generator = None
    SYSTEM_PROMPTS = {
        "general": "You are a helpful assistant. Provide information based on the context provided."
    }


# Check overall system status
def get_system_status():
    """Get overall system status based on component availability."""
    components = {
        "confluence": confluence_client is not None,
        "remedy": remedy_client is not None,
        "embedding": embedding_processor is not None,
        "index": index_manager is not None,
        "vector_search": vector_retriever is not None,
        "lexical_search": lexical_retriever is not None,
        "hybrid_search": hybrid_retriever is not None,
        "llm": gemini_client is not None and rag_generator is not None
    }
    
    # Calculate status based on critical components
    critical_components = ["embedding", "index", "llm"]
    critical_status = all(components.get(c, False) for c in critical_components)
    
    # Count available data sources
    data_sources = sum(1 for s in ["confluence", "remedy"] if components.get(s, False))
    
    # Count available search methods
    search_methods = sum(1 for s in ["vector_search", "lexical_search", "hybrid_search"] if components.get(s, False))
    
    if critical_status and data_sources > 0 and search_methods > 0:
        status = "ok"
    elif critical_status and (data_sources > 0 or search_methods > 0):
        status = "degraded"
    else:
        status = "error"
    
    return status, components


@api_bp.route('/health', methods=['GET'])
@timer_decorator
def health_check():
    """Health check endpoint."""
    try:
        # Get system status
        status, components = get_system_status()
        
        # Check data source connections
        confluence_status = False
        remedy_status = False
        
        if confluence_client:
            try:
                spaces = confluence_client.get_spaces(limit=1)
                confluence_status = len(spaces) > 0
            except Exception as e:
                logger.error(f"Confluence health check failed: {e}")
        
        if remedy_client:
            try:
                token = remedy_client.get_token()
                remedy_status = token is not None
            except Exception as e:
                logger.error(f"Remedy health check failed: {e}")
        
        # Get index stats
        index_stats = {}
        try:
            # Add vector index stats if available
            if index_manager and hasattr(index_manager.index, "ntotal"):
                vector_size = index_manager.index.ntotal
                index_stats["vector_index_size"] = vector_size
            
            # Add lexical index stats if available
            if lexical_retriever and hasattr(lexical_retriever, "doc_tokens"):
                lexical_size = len(lexical_retriever.doc_tokens)
                index_stats["lexical_index_size"] = lexical_size
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
        
        # Build response
        response = {
            "status": status,
            "version": "1.0.0",
            "endpoints": [
                "/api/health",
                "/api/query",
                "/api/chat",
                "/api/index"
            ],
            "sources": {
                "confluence": confluence_status,
                "remedy": remedy_status
            },
            "components": components,
            "index_stats": index_stats
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "version": "1.0.0",
            "error": str(e)
        }), 500


@api_bp.route('/query', methods=['POST'])
@timer_decorator
def query():
    """Query endpoint for RAG."""
    try:
        # Parse request
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "No request data provided"}), 400
        
        # Validate request data with minimal validation
        query_text = req_data.get("query")
        if not query_text:
            return jsonify({"error": "No query text provided"}), 400
        
        # Extract other parameters with defaults
        sources = req_data.get("sources", ["confluence", "remedy"])
        top_k = int(req_data.get("top_k", config.NUM_RESULTS))
        search_mode = req_data.get("mode", "hybrid")
        multi_query = req_data.get("multi_query", False)
        rerank_results = req_data.get("rerank", True)
        prompt_template = req_data.get("prompt_template")
        temperature = float(req_data.get("temperature", config.TEMPERATURE))
        max_tokens = int(req_data.get("max_tokens", 1024))
        stream_response = req_data.get("stream", False)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        logger.info(f"Query received: {query_text[:100]}...")
        
        # Check system status
        status, components = get_system_status()
        if status == "error":
            return jsonify({
                "answer": "The system is currently unavailable. Please try again later.",
                "sources": [],
                "query": query_text,
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Select retriever based on mode and availability
        retriever = None
        if search_mode == "vector" and components.get("vector_search"):
            retriever = vector_retriever
        elif search_mode == "lexical" and components.get("lexical_search"):
            retriever = lexical_retriever
        elif components.get("hybrid_search"):
            retriever = hybrid_retriever
        elif components.get("vector_search"):
            retriever = vector_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to vector search")
        elif components.get("lexical_search"):
            retriever = lexical_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to lexical search")
        else:
            return jsonify({
                "answer": "No search retrievers are currently available. Please try again later.",
                "sources": [],
                "query": query_text,
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Retrieve documents
        retrieval_start = time.time()
        
        results = []
        try:
            search_func = retriever.multi_query_search if multi_query else retriever.search
            results = search_func(query_text, k=top_k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            traceback.print_exc()
        
        retrieval_time = time.time() - retrieval_start
        timings["retrieval"] = retrieval_time
        
        # Rerank if requested and available
        if rerank_results and results and reranker and search_mode != "lexical":
            try:
                rerank_start = time.time()
                results = reranker.rerank(query_text, results, k=top_k)
                timings["reranking"] = time.time() - rerank_start
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
        
        # Generate response if LLM is available
        if results and rag_generator:
            # Get system prompt based on template
            system_prompt = None
            if prompt_template and prompt_template in SYSTEM_PROMPTS:
                system_prompt = SYSTEM_PROMPTS.get(prompt_template)
            
            # Generate answer
            generation_start = time.time()
            
            if stream_response:
                # Stream the response
                def generate():
                    try:
                        # First, send a JSON object with metadata
                        source_info = []
                        for result in results:
                            metadata = result.get("metadata", {})
                            source_info.append({
                                "title": metadata.get("title", ""),
                                "source": metadata.get("source", "Unknown"),
                                "url": metadata.get("source_url", ""),
                                "score": result.get("score", 0)
                            })
                        
                        metadata = {
                            "query": query_text,
                            "sources": source_info,
                            "timing": {
                                "retrieval": retrieval_time
                            }
                        }
                        
                        yield json.dumps({"metadata": metadata}) + "\n"
                        
                        # Stream the answer
                        try:
                            response_stream = rag_generator.generate(
                                query=query_text,
                                chunks=results,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=True
                            )
                            
                            for chunk in response_stream:
                                if hasattr(chunk, "text") and chunk.text:
                                    yield json.dumps({"content": chunk.text}) + "\n"
                        except Exception as e:
                            logger.error(f"Error streaming response: {e}")
                            yield json.dumps({"content": f"Error generating response: {str(e)}"}) + "\n"
                    
                    except Exception as e:
                        logger.error(f"Error in generator: {e}")
                        yield json.dumps({"content": "Error generating response."}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Generate a complete response
                try:
                    answer = rag_generator.generate(
                        query=query_text,
                        chunks=results,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    generation_time = time.time() - generation_start
                    timings["generation"] = generation_time
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    answer = f"Error generating response. The system encountered a problem: {str(e)}"
                
                # Prepare sources for response
                sources_info = []
                for result in results:
                    metadata = result.get("metadata", {})
                    sources_info.append({
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", "Unknown"),
                        "url": metadata.get("source_url", ""),
                        "score": result.get("score", 0)
                    })
                
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = {
                    "answer": answer,
                    "sources": sources_info,
                    "query": query_text,
                    "timing": timings
                }
                
                return jsonify(response)
        else:
            # No results found or LLM unavailable
            if not results:
                msg = "I couldn't find any relevant information to answer your question."
            else:
                msg = "Unable to generate a response at this time. The language model is currently unavailable."
            
            if stream_response:
                def generate():
                    metadata = {
                        "query": query_text,
                        "sources": [],
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    yield json.dumps({"content": msg}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = {
                    "answer": msg,
                    "sources": [],
                    "query": query_text,
                    "timing": timings
                }
                
                return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_bp.route('/chat', methods=['POST'])
@timer_decorator
def chat():
    """Chat endpoint for RAG."""
    try:
        # Parse request with error handling
        try:
            req_data = request.get_json()
            if not req_data:
                return jsonify({"error": "No request data provided"}), 400
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Validate messages field
        messages = req_data.get("messages")
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Extract other parameters with defaults
        sources = req_data.get("sources", ["confluence", "remedy"])
        top_k = int(req_data.get("top_k", config.NUM_RESULTS))
        search_mode = req_data.get("mode", "hybrid")
        multi_query = req_data.get("multi_query", False)
        rerank_results = req_data.get("rerank", True)
        prompt_template = req_data.get("prompt_template")
        temperature = float(req_data.get("temperature", config.TEMPERATURE))
        max_tokens = int(req_data.get("max_tokens", 1024))
        stream_response = req_data.get("stream", False)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        # Check system status
        status, components = get_system_status()
        if status == "error":
            return jsonify({
                "message": {"role": "assistant", "content": "The system is currently unavailable. Please try again later."},
                "sources": [],
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Extract query from last user message
        query_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                query_text = message.get("content", "")
                break
        
        if not query_text:
            return jsonify({"error": "No user message found"}), 400
        
        logger.info(f"Chat query received: {query_text[:100]}...")
        
        # Select retriever based on mode and availability
        retriever = None
        if search_mode == "vector" and components.get("vector_search"):
            retriever = vector_retriever
        elif search_mode == "lexical" and components.get("lexical_search"):
            retriever = lexical_retriever
        elif components.get("hybrid_search"):
            retriever = hybrid_retriever
        elif components.get("vector_search"):
            retriever = vector_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to vector search")
        elif components.get("lexical_search"):
            retriever = lexical_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to lexical search")
        else:
            return jsonify({
                "message": {"role": "assistant", "content": "No search retrievers a












"""
Configuration module for the RAG application.
Updated with comprehensive error handling and robust defaults.
"""
import os
import sys
import tempfile
from pathlib import Path
import logging

# Set up basic logging for early errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
basic_logger = logging.getLogger("config")

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    basic_logger.info("Loaded environment variables from .env file")
except ImportError:
    basic_logger.warning("dotenv package not available, using OS environment variables only")
except Exception as e:
    basic_logger.error(f"Error loading .env file: {e}")

# Base directory of the application
try:
    BASE_DIR = Path(__file__).resolve().parent
except Exception as e:
    basic_logger.error(f"Error determining base directory: {e}")
    BASE_DIR = Path(os.getcwd())

# Flask configuration with safe defaults
try:
    DEBUG = os.getenv("FLASK_ENV") == "development"
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24).hex())
    PORT = int(os.getenv("PORT", 5000))
except Exception as e:
    basic_logger.error(f"Error setting Flask configuration: {e}")
    DEBUG = True
    SECRET_KEY = os.urandom(24).hex()
    PORT = 5000

# API settings with safe fallbacks
try:
    # Confluence API settings
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")
    CONFLUENCE_SPACE_ID = os.getenv("CONFLUENCE_SPACE_ID", "")
    CONFLUENCE_USER_ID = os.getenv("CONFLUENCE_USER_ID", "")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")

    # Remedy API settings
    REMEDY_SERVER = os.getenv("REMEDY_SERVER", "")
    REMEDY_API_BASE = os.getenv("REMEDY_API_BASE", "")
    REMEDY_USERNAME = os.getenv("REMEDY_USERNAME", "")
    REMEDY_PASSWORD = os.getenv("REMEDY_PASSWORD", "")

    # Google Cloud and Gemini settings
    PROJECT_ID = os.getenv("PROJECT_ID", "")
    REGION = os.getenv("REGION", "us-central1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")
except Exception as e:
    basic_logger.error(f"Error loading API settings: {e}")
    # Minimal defaults to prevent crashes
    CONFLUENCE_URL = ""
    CONFLUENCE_SPACE_ID = ""
    CONFLUENCE_USER_ID = ""
    CONFLUENCE_API_TOKEN = ""
    REMEDY_SERVER = ""
    REMEDY_API_BASE = ""
    REMEDY_USERNAME = ""
    REMEDY_PASSWORD = ""
    PROJECT_ID = ""
    REGION = "us-central1"
    MODEL_NAME = "gemini-1.5-pro"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Cache settings with robust directory creation
try:
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "rag_cache")
    Path(CACHE_DIR).mkdir(exist_ok=True)
    basic_logger.info(f"Cache directory set to {CACHE_DIR}")
except Exception as e:
    basic_logger.error(f"Error creating cache directory: {e}")
    CACHE_DIR = tempfile.gettempdir()

# Vector store settings
try:
    VECTOR_STORE_PATH = os.path.join(CACHE_DIR, "vector_store")
    Path(VECTOR_STORE_PATH).mkdir(exist_ok=True)
except Exception as e:
    basic_logger.error(f"Error creating vector store directory: {e}")
    VECTOR_STORE_PATH = CACHE_DIR

# Embedding model settings - with multiple fallback options
try:
    # Primary model - complex but better quality
    EMBEDDING_MODEL_PRIMARY = "all-MiniLM-L6-v2"
    # Fallback model - simpler and more compatible
    EMBEDDING_MODEL_FALLBACK = "paraphrase-MiniLM-L3-v2"
    # Default to fallback for stability
    EMBEDDING_MODEL = EMBEDDING_MODEL_FALLBACK
    EMBEDDING_DIMENSION = 384  # Dimension for the embedding model chosen
except Exception as e:
    basic_logger.error(f"Error setting embedding model configuration: {e}")
    EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
    EMBEDDING_DIMENSION = 384

# RAG settings with safe defaults
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))
    NUM_RESULTS = int(os.getenv("NUM_RESULTS", 5))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
except (ValueError, TypeError) as e:
    basic_logger.error(f"Error parsing RAG settings: {e}")
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    NUM_RESULTS = 5
    TEMPERATURE = 0.2

# Try to import loguru for better logging
try:
    from loguru import logger
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
    )
    
    # Add file handler
    logs_dir = os.path.join(BASE_DIR, "logs")
    Path(logs_dir).mkdir(exist_ok=True)
    log_file = os.path.join(logs_dir, f"{Path(__file__).stem}.log")
    
    logger.add(
        log_file,
        level=LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8"
    )
    
    logger.info("Loguru logger configured")
    logging_initialized = True
except ImportError:
    basic_logger.warning("Loguru not available, using basic logging")
    logging_initialized = False
except Exception as e:
    basic_logger.error(f"Error configuring Loguru: {e}")
    logging_initialized = False

# Dictionary of operational status
SYSTEM_STATUS = {
    "config_loaded": True,
    "logging_initialized": logging_initialized,
    "cache_available": os.path.exists(CACHE_DIR) and os.access(CACHE_DIR, os.W_OK)
}

def validate_config():
    """
    Validate critical configuration settings with robust error handling.
    
    Returns:
        bool: True if critical config is valid, False otherwise
    """
    try:
        # Define required vars with relaxed requirements
        # Only consider completely missing values as invalid
        required_vars = {
            "CONFLUENCE_URL": bool(CONFLUENCE_URL),
            "CONFLUENCE_API_TOKEN": bool(CONFLUENCE_API_TOKEN),
            "REMEDY_API_BASE": bool(REMEDY_API_BASE),
            "PROJECT_ID": bool(PROJECT_ID),
            "MODEL_NAME": bool(MODEL_NAME)
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            missing_vars_str = ", ".join(missing_vars)
            if logging_initialized:
                logger.warning(f"Missing environment variables: {missing_vars_str}")
            else:
                basic_logger.warning(f"Missing environment variables: {missing_vars_str}")
            
        # System can run even with missing variables in degraded mode
        SYSTEM_STATUS["config_valid"] = len(missing_vars) <= 2  # Allow up to 2 missing vars
        return SYSTEM_STATUS["config_valid"]
            
    except Exception as e:
        error_msg = f"Error validating configuration: {e}"
        if logging_initialized:
            logger.error(error_msg)
        else:
            basic_logger.error(error_msg)
        SYSTEM_STATUS["config_valid"] = False
        return False

# Validate on import
validate_config()

# Log completion
if logging_initialized:
    logger.info("Configuration loaded and validated")
else:
    basic_logger.info("Configuration loaded and validated")





"""
API routes for the RAG application.
Updated with comprehensive error handling to prevent crashes.
"""
from typing import List, Dict, Any, Optional, Union, Generator
import time
import json
import traceback
from flask import Blueprint, request, jsonify, Response, stream_with_context, current_app

import config
from utils import get_logger, timer_decorator

# Initialize logger
logger = get_logger("api_routes")

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize components with robust error handling
try:
    from modules.data_sources import ConfluenceClient, ConfluenceContentProcessor
    confluence_client = ConfluenceClient()
    confluence_processor = ConfluenceContentProcessor(confluence_client)
    logger.info("Confluence components initialized")
except Exception as e:
    logger.error(f"Error initializing Confluence components: {e}")
    confluence_client = None
    confluence_processor = None

try:
    from modules.data_sources import RemedyClient, RemedyContentProcessor
    remedy_client = RemedyClient()
    remedy_processor = RemedyContentProcessor(remedy_client)
    logger.info("Remedy components initialized")
except Exception as e:
    logger.error(f"Error initializing Remedy components: {e}")
    remedy_client = None
    remedy_processor = None

try:
    from modules.processing import EmbeddingProcessor
    embedding_processor = EmbeddingProcessor()
    logger.info("Embedding processor initialized")
except Exception as e:
    logger.error(f"Error initializing Embedding processor: {e}")
    embedding_processor = None

try:
    from modules.processing import IndexManager
    index_manager = IndexManager()
    logger.info("Index manager initialized")
except Exception as e:
    logger.error(f"Error initializing Index manager: {e}")
    index_manager = None

try:
    from modules.retrieval import VectorSearchRetriever
    vector_retriever = VectorSearchRetriever(embedding_processor, index_manager)
    logger.info("Vector retriever initialized")
except Exception as e:
    logger.error(f"Error initializing Vector retriever: {e}")
    vector_retriever = None

try:
    from modules.retrieval import LexicalSearchRetriever
    lexical_retriever = LexicalSearchRetriever()
    logger.info("Lexical retriever initialized")
except Exception as e:
    logger.error(f"Error initializing Lexical retriever: {e}")
    lexical_retriever = None

try:
    from modules.retrieval import HybridSearchRetriever, ReRanker
    hybrid_retriever = None
    if vector_retriever and lexical_retriever:
        hybrid_retriever = HybridSearchRetriever(vector_retriever, lexical_retriever)
        logger.info("Hybrid retriever initialized")
    reranker = ReRanker(vector_retriever) if vector_retriever else None
    logger.info("Reranker initialized")
except Exception as e:
    logger.error(f"Error initializing Hybrid retriever or Reranker: {e}")
    hybrid_retriever = None
    reranker = None

try:
    from modules.llm import GeminiClient, RAGGenerator, SYSTEM_PROMPTS
    gemini_client = GeminiClient()
    rag_generator = RAGGenerator(gemini_client)
    logger.info("LLM components initialized")
except Exception as e:
    logger.error(f"Error initializing LLM components: {e}")
    gemini_client = None
    rag_generator = None
    SYSTEM_PROMPTS = {
        "general": "You are a helpful assistant. Provide information based on the context provided."
    }


# Check overall system status
def get_system_status():
    """Get overall system status based on component availability."""
    components = {
        "confluence": confluence_client is not None,
        "remedy": remedy_client is not None,
        "embedding": embedding_processor is not None,
        "index": index_manager is not None,
        "vector_search": vector_retriever is not None,
        "lexical_search": lexical_retriever is not None,
        "hybrid_search": hybrid_retriever is not None,
        "llm": gemini_client is not None and rag_generator is not None
    }
    
    # Calculate status based on critical components
    critical_components = ["embedding", "index", "llm"]
    critical_status = all(components.get(c, False) for c in critical_components)
    
    # Count available data sources
    data_sources = sum(1 for s in ["confluence", "remedy"] if components.get(s, False))
    
    # Count available search methods
    search_methods = sum(1 for s in ["vector_search", "lexical_search", "hybrid_search"] if components.get(s, False))
    
    if critical_status and data_sources > 0 and search_methods > 0:
        status = "ok"
    elif critical_status and (data_sources > 0 or search_methods > 0):
        status = "degraded"
    else:
        status = "error"
    
    return status, components


@api_bp.route('/health', methods=['GET'])
@timer_decorator
def health_check():
    """Health check endpoint."""
    try:
        # Get system status
        status, components = get_system_status()
        
        # Check data source connections
        confluence_status = False
        remedy_status = False
        
        if confluence_client:
            try:
                spaces = confluence_client.get_spaces(limit=1)
                confluence_status = len(spaces) > 0
            except Exception as e:
                logger.error(f"Confluence health check failed: {e}")
        
        if remedy_client:
            try:
                token = remedy_client.get_token()
                remedy_status = token is not None
            except Exception as e:
                logger.error(f"Remedy health check failed: {e}")
        
        # Get index stats
        index_stats = {}
        try:
            # Add vector index stats if available
            if index_manager and hasattr(index_manager.index, "ntotal"):
                vector_size = index_manager.index.ntotal
                index_stats["vector_index_size"] = vector_size
            
            # Add lexical index stats if available
            if lexical_retriever and hasattr(lexical_retriever, "doc_tokens"):
                lexical_size = len(lexical_retriever.doc_tokens)
                index_stats["lexical_index_size"] = lexical_size
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
        
        # Build response
        response = {
            "status": status,
            "version": "1.0.0",
            "endpoints": [
                "/api/health",
                "/api/query",
                "/api/chat",
                "/api/index"
            ],
            "sources": {
                "confluence": confluence_status,
                "remedy": remedy_status
            },
            "components": components,
            "index_stats": index_stats
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "version": "1.0.0",
            "error": str(e)
        }), 500


@api_bp.route('/query', methods=['POST'])
@timer_decorator
def query():
    """Query endpoint for RAG."""
    try:
        # Parse request
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "No request data provided"}), 400
        
        # Validate request data with minimal validation
        query_text = req_data.get("query")
        if not query_text:
            return jsonify({"error": "No query text provided"}), 400
        
        # Extract other parameters with defaults
        sources = req_data.get("sources", ["confluence", "remedy"])
        top_k = int(req_data.get("top_k", config.NUM_RESULTS))
        search_mode = req_data.get("mode", "hybrid")
        multi_query = req_data.get("multi_query", False)
        rerank_results = req_data.get("rerank", True)
        prompt_template = req_data.get("prompt_template")
        temperature = float(req_data.get("temperature", config.TEMPERATURE))
        max_tokens = int(req_data.get("max_tokens", 1024))
        stream_response = req_data.get("stream", False)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        logger.info(f"Query received: {query_text[:100]}...")
        
        # Check system status
        status, components = get_system_status()
        if status == "error":
            return jsonify({
                "answer": "The system is currently unavailable. Please try again later.",
                "sources": [],
                "query": query_text,
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Select retriever based on mode and availability
        retriever = None
        if search_mode == "vector" and components.get("vector_search"):
            retriever = vector_retriever
        elif search_mode == "lexical" and components.get("lexical_search"):
            retriever = lexical_retriever
        elif components.get("hybrid_search"):
            retriever = hybrid_retriever
        elif components.get("vector_search"):
            retriever = vector_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to vector search")
        elif components.get("lexical_search"):
            retriever = lexical_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to lexical search")
        else:
            return jsonify({
                "answer": "No search retrievers are currently available. Please try again later.",
                "sources": [],
                "query": query_text,
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Retrieve documents
        retrieval_start = time.time()
        
        results = []
        try:
            search_func = retriever.multi_query_search if multi_query else retriever.search
            results = search_func(query_text, k=top_k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            traceback.print_exc()
        
        retrieval_time = time.time() - retrieval_start
        timings["retrieval"] = retrieval_time
        
        # Rerank if requested and available
        if rerank_results and results and reranker and search_mode != "lexical":
            try:
                rerank_start = time.time()
                results = reranker.rerank(query_text, results, k=top_k)
                timings["reranking"] = time.time() - rerank_start
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
        
        # Generate response if LLM is available
        if results and rag_generator:
            # Get system prompt based on template
            system_prompt = None
            if prompt_template and prompt_template in SYSTEM_PROMPTS:
                system_prompt = SYSTEM_PROMPTS.get(prompt_template)
            
            # Generate answer
            generation_start = time.time()
            
            if stream_response:
                # Stream the response
                def generate():
                    try:
                        # First, send a JSON object with metadata
                        source_info = []
                        for result in results:
                            metadata = result.get("metadata", {})
                            source_info.append({
                                "title": metadata.get("title", ""),
                                "source": metadata.get("source", "Unknown"),
                                "url": metadata.get("source_url", ""),
                                "score": result.get("score", 0)
                            })
                        
                        metadata = {
                            "query": query_text,
                            "sources": source_info,
                            "timing": {
                                "retrieval": retrieval_time
                            }
                        }
                        
                        yield json.dumps({"metadata": metadata}) + "\n"
                        
                        # Stream the answer
                        try:
                            response_stream = rag_generator.generate(
                                query=query_text,
                                chunks=results,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=True
                            )
                            
                            for chunk in response_stream:
                                if hasattr(chunk, "text") and chunk.text:
                                    yield json.dumps({"content": chunk.text}) + "\n"
                        except Exception as e:
                            logger.error(f"Error streaming response: {e}")
                            yield json.dumps({"content": f"Error generating response: {str(e)}"}) + "\n"
                    
                    except Exception as e:
                        logger.error(f"Error in generator: {e}")
                        yield json.dumps({"content": "Error generating response."}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Generate a complete response
                try:
                    answer = rag_generator.generate(
                        query=query_text,
                        chunks=results,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    generation_time = time.time() - generation_start
                    timings["generation"] = generation_time
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    answer = f"Error generating response. The system encountered a problem: {str(e)}"
                
                # Prepare sources for response
                sources_info = []
                for result in results:
                    metadata = result.get("metadata", {})
                    sources_info.append({
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", "Unknown"),
                        "url": metadata.get("source_url", ""),
                        "score": result.get("score", 0)
                    })
                
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = {
                    "answer": answer,
                    "sources": sources_info,
                    "query": query_text,
                    "timing": timings
                }
                
                return jsonify(response)
        else:
            # No results found or LLM unavailable
            if not results:
                msg = "I couldn't find any relevant information to answer your question."
            else:
                msg = "Unable to generate a response at this time. The language model is currently unavailable."
            
            if stream_response:
                def generate():
                    metadata = {
                        "query": query_text,
                        "sources": [],
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    yield json.dumps({"content": msg}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = {
                    "answer": msg,
                    "sources": [],
                    "query": query_text,
                    "timing": timings
                }
                
                return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_bp.route('/chat', methods=['POST'])
@timer_decorator
def chat():
    """Chat endpoint for RAG."""
    try:
        # Parse request with error handling
        try:
            req_data = request.get_json()
            if not req_data:
                return jsonify({"error": "No request data provided"}), 400
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Validate messages field
        messages = req_data.get("messages")
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Extract other parameters with defaults
        sources = req_data.get("sources", ["confluence", "remedy"])
        top_k = int(req_data.get("top_k", config.NUM_RESULTS))
        search_mode = req_data.get("mode", "hybrid")
        multi_query = req_data.get("multi_query", False)
        rerank_results = req_data.get("rerank", True)
        prompt_template = req_data.get("prompt_template")
        temperature = float(req_data.get("temperature", config.TEMPERATURE))
        max_tokens = int(req_data.get("max_tokens", 1024))
        stream_response = req_data.get("stream", False)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        # Check system status
        status, components = get_system_status()
        if status == "error":
            return jsonify({
                "message": {"role": "assistant", "content": "The system is currently unavailable. Please try again later."},
                "sources": [],
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Extract query from last user message
        query_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                query_text = message.get("content", "")
                break
        
        if not query_text:
            return jsonify({"error": "No user message found"}), 400
        
        logger.info(f"Chat query received: {query_text[:100]}...")
        
        # Select retriever based on mode and availability
        retriever = None
        if search_mode == "vector" and components.get("vector_search"):
            retriever = vector_retriever
        elif search_mode == "lexical" and components.get("lexical_search"):
            retriever = lexical_retriever
        elif components.get("hybrid_search"):
            retriever = hybrid_retriever
        elif components.get("vector_search"):
            retriever = vector_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to vector search")
        elif components.get("lexical_search"):
            retriever = lexical_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to lexical search")
        else:
            return jsonify({
                "message": {"role": "assistant", "content": "No search retrievers a





retriever = None
        if search_mode == "vector" and components.get("vector_search"):
            retriever = vector_retriever
        elif search_mode == "lexical" and components.get("lexical_search"):
            retriever = lexical_retriever
        elif components.get("hybrid_search"):
            retriever = hybrid_retriever
        elif components.get("vector_search"):
            retriever = vector_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to vector search")
        elif components.get("lexical_search"):
            retriever = lexical_retriever
            logger.warning(f"Requested mode {search_mode} not available, falling back to lexical search")
        else:
            return jsonify({
                "message": {"role": "assistant", "content": "No search retrievers are currently available. Please try again later."},
                "sources": [],
                "timing": {"total": time.time() - overall_start},
                "status": "error"
            }), 503
        
        # Retrieve documents
        retrieval_start = time.time()
        
        results = []
        try:
            search_func = retriever.multi_query_search if multi_query else retriever.search
            results = search_func(query_text, k=top_k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
        
        retrieval_time = time.time() - retrieval_start
        timings["retrieval"] = retrieval_time
        
        # Rerank if requested and available
        if rerank_results and results and reranker and search_mode != "lexical":
            try:






try:
            search_func = retriever.multi_query_search if multi_query else retriever.search
            results = search_func(query_text, k=top_k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
        
        retrieval_time = time.time() - retrieval_start
        timings["retrieval"] = retrieval_time
        
        # Rerank if requested and available
        if rerank_results and results and reranker and search_mode != "lexical":
            try:
                rerank_start = time.time()
                results = reranker.rerank(query_text, results, k=top_k)
                timings["reranking"] = time.time() - rerank_start
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
        
        # Generate response if LLM is available
        if results and rag_generator:
            # Get system prompt based on template
            system_prompt = None
            if prompt_template and prompt_template in SYSTEM_PROMPTS:
                system_prompt = SYSTEM_PROMPTS.get(prompt_template)
            
            # Convert messages to format expected by generator
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Generate answer
            generation_start = time.time()







generation_start = time.time()
            
            if stream_response:
                # Stream the response
                def generate():
                    try:
                        # First, send a JSON object with metadata
                        source_info = []
                        for result in results:
                            metadata = result.get("metadata", {})
                            source_info.append({
                                "title": metadata.get("title", ""),
                                "source": metadata.get("source", "Unknown"),
                                "url": metadata.get("source_url", ""),
                                "score": result.get("score", 0)
                            })
                        
                        metadata = {
                            "query": query_text,
                            "sources": source_info,
                            "timing": {
                                "retrieval": retrieval_time
                            }
                        }
                        
                        yield json.dumps({"metadata": metadata}) + "\n"
                        
                        # Stream the answer
                        try:
                            response_stream = rag_generator.generate_chat(
                                messages=formatted_messages,
                                chunks=results,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=True
                            )
                            




for chunk in response_stream:
                                if hasattr(chunk, "text") and chunk.text:
                                    yield json.dumps({"content": chunk.text}) + "\n"
                        except Exception as e:
                            logger.error(f"Error streaming chat response: {e}")
                            yield json.dumps({"content": f"Error generating response: {str(e)}"}) + "\n"
                    
                    except Exception as e:
                        logger.error(f"Error in chat generator: {e}")
                        yield json.dumps({"content": "Error generating response."}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Generate a complete response
                try:
                    answer = rag_generator.generate_chat(
                        messages=formatted_messages,
                        chunks=results,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    generation_time = time.time() - generation_start
                    timings["generation"] = generation_time
                    
                except Exception as e:
                    logger.error(f"Error generating chat response: {e}")
                    answer = f"Error generating response. The system encountered a problem: {str(e)}"
                
                # Prepare sources for response
                sources_info = []
                for result in results:
                    metadata = result.get("metadata", {})
                    sources_info.append({
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", "Unknown"),
                        "url": metadata.get("source_url", ""),
                        "score": result.get("score", 0)
                    })






Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = {
                    "message": {"role": "assistant", "content": answer},
                    "sources": sources_info,
                    "timing": timings
                }
                
                return jsonify(response)
        else:
            # No results found or LLM unavailable
            if not results:
                msg = "I couldn't find any relevant information to answer your question."
            else:
                msg = "Unable to generate a response at this time. The language model is currently unavailable."
            
            if stream_response:
                def generate():
                    metadata = {
                        "query": query_text,
                        "sources": [],
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    yield json.dumps({"content": msg}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response







Build response
                response = {
                    "message": {"role": "assistant", "content": msg},
                    "sources": [],
                    "timing": timings
                }
                
                return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_bp.route('/index', methods=['POST'])
@timer_decorator
def index():
    """Index data from sources."""
    try:
        # Parse request with basic validation
        try:
            req_data = request.get_json()
            if not req_data:
                return jsonify({"error": "No request data provided"}), 400
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Extract parameters with defaults
        sources = req_data.get("sources", ["confluence", "remedy"])
        confluence_space_key = req_data.get("confluence_space_key")
        limit = req_data.get("limit")
        chunking_strategy = req_data.get("chunking_strategy", "semantic")
        force_reindex = req_data.get("force_reindex", False)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        






Check component availability
        if not embedding_processor or not index_manager:
            return jsonify({
                "status": "error",
                "message": "Indexing components are not available",
                "counts": {},
                "timing": {"total": time.time() - overall_start}
            }), 503
        
        # Create chunker
        try:
            from modules.processing import ChunkerFactory
            chunker = ChunkerFactory.get_chunker(
                strategy=chunking_strategy,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            logger.info(f"Created chunker with strategy: {chunking_strategy}")
        except Exception as e:
            logger.error(f"Error creating chunker: {e}")
            return jsonify({
                "status": "error",
                "message": f"Failed to create chunker: {str(e)}",
                "counts": {},
                "timing": {"total": time.time() - overall_start}
            }), 500
        
        # Clear indexes if requested
        if force_reindex:
            try:
                if index_manager:
                    index_manager.clear_index()
                if lexical_retriever:
                    lexical_retriever.clear_index()
                logger.info("Cleared existing indexes")
            except Exception as e:
                logger.error(f"Error clearing indexes: {e}")
        
        # Initialize document counts
        document_counts = {}









Initialize document counts
        document_counts = {}
        all_chunks = []
        
        # Process Confluence data
        if "confluence" in sources and confluence_processor:
            confluence_start = time.time()
            
            # Get Confluence pages
            try:
                if confluence_space_key:
                    # Get pages from specific space
                    pages = confluence_processor.process_space(
                        space_key=confluence_space_key,
                        limit=limit
                    )
                else:
                    # Get pages from configured space
                    pages = confluence_processor.process_space(
                        space_key=config.CONFLUENCE_SPACE_ID,
                        limit=limit
                    )
                
                # Create documents for indexing
                confluence_docs = []
                for page in pages:
                    doc = confluence_processor.create_document_for_indexing(page)
                    if doc:
                        confluence_docs.append(doc)
                
                # Chunk documents
                confluence_chunks = []
                for doc in confluence_docs:
                    chunks = chunker.chunk_document(doc)
                    confluence_chunks.extend(chunks)
                
                # Add to all chunks
                all_chunks.extend(confluence_chunks)
                document_counts["confluence"] = len(confluence_docs)
                
                timings["confluence"] = time.time() - confluence_start
                logger.info(f"Processed {len(confluence_docs)} Confluence documents into {len(confluence_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing Confluence data: {e}")
                timings["confluence"] = time.time() - confluence_start
                document_counts["confluence"] = 0
        
        # Process Remedy data
        if "remedy" in sources and remedy_processor:
            remedy_start = time.time()
            
            try:
                # Get Remedy data (incidents, changes, knowledge articles)
                remedy_docs = []








remedy_docs = []
                
                # Get incidents
                if remedy_client:
                    try:
                        incidents = remedy_client.search_incidents(limit=limit or 100)
                        for incident in incidents:
                            incident_id = incident.get("id")
                            if incident_id:
                                processed = remedy_processor.process_incident(incident_id)
                                if processed:
                                    doc = remedy_processor.create_document_for_indexing(processed, "incident")
                                    if doc:
                                        remedy_docs.append(doc)
                        
                        # Get change requests
                        changes = remedy_client.search_change_requests(limit=limit or 100)
                        for change in changes:
                            change_id = change.get("id")
                            if change_id:
                                processed = remedy_processor.process_change_request(change_id)
                                if processed:
                                    doc = remedy_processor.create_document_for_indexing(processed, "change")
                                    if doc:
                                        remedy_docs.append(doc)
                        
                        # Get knowledge articles
                        articles = remedy_client.search_knowledge_articles(limit=limit or 100)
                        for article in articles:
                            article_id = article.get("id")
                            if article_id:
                                processed = remedy_processor.process_knowledge_article(article_id)
                                if processed:
                                    doc = remedy_processor.create_document_for_indexing(processed, "knowledge")
                                    if doc:
                                        remedy_docs.append(doc)
                    except Exception as e:
                        logger.error(f"Error retrieving Remedy data: {e}")
                
                # Chunk documents
                remedy_chunks = []
                for doc in remedy_docs:
                    chunks = chunker.chunk_document(doc)
                    remedy_chunks.extend(chunks)
                
                # Add to all ch




Add to all chunks
                all_chunks.extend(remedy_chunks)
                document_counts["remedy"] = len(remedy_docs)
                
                timings["remedy"] = time.time() - remedy_start
                logger.info(f"Processed {len(remedy_docs)} Remedy documents into {len(remedy_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing Remedy data: {e}")
                timings["remedy"] = time.time() - remedy_start
                document_counts["remedy"] = 0
        
        # Index all chunks
        if all_chunks:
            indexing_start = time.time()
            
            # Index for vector search
            try:
                if vector_retriever:
                    vector_retriever.index_chunks(all_chunks)
                    logger.info(f"Indexed {len(all_chunks)} chunks for vector search")
            except Exception as e:
                logger.error(f"Error indexing for vector search: {e}")
            
            # Index for lexical search
            try:
                if lexical_retriever:
                    lexical_retriever.index_chunks(all_chunks)
                    logger.info(f"Indexed {len(all_chunks)} chunks for lexical search")
            except Exception as e:
                logger.error(f"Error indexing for lexical search: {e}")
            
            timings["indexing"] = time.time() - indexing_start
        
        # Calculate total time
        total_time = time.time() - overall_start
        timings["total"] = total_time
        
        # Build response
        document_counts["total"] = sum(document_counts.values())
        document_counts["chunks"] = len(all_chunks)
        
        response = {
            "status": "success",
            "message": f"Indexed {document_counts.get('total', 0)} documents with {len(all_chunks)} chunks",
            "counts": document_counts,
            "timing": timings
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "counts": {},
            "timing": {"total": time.time() - time.time()}
        }), 500





                   
        document_counts["total"] = sum(document_counts.values())
        document_counts["chunks"] = len(all_chunks)
        
        response = {
            "status": "success",
            "message": f"Indexed {document_counts.get('total', 0)} documents with {len(all_chunks)} chunks",
            "counts": document_counts,
            "timing": timings
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "counts": {},
            "timing": {"total": time.time() - time.time()}
        }), 500