"""
Configuration file containing all constants and configuration items
"""
import os

# 1. File paths
VECTOR_STORE_PATH = "faiss_index"
HISTORY_FILE = "chat_history.json"

# 2. Model configuration (OpenAI only)
DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini",
]

# OpenAI credentials & endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# Note: OLLAMA_BASE_URL kept for backward compatibility with embedding models
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding models (sentence-transformers - open source, no API key needed)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dimensions - best quality
AVAILABLE_EMBEDDING_MODELS = [
    "BAAI/bge-large-en-v1.5",  # 1024 dim - Best quality
    "BAAI/bge-base-en-v1.5",   # 768 dim - Good balance
    "all-mpnet-base-v2",        # 768 dim - General purpose
    "all-MiniLM-L6-v2"         # 384 dim - Fast, lightweight
]
EMBEDDING_BASE_URL = "http://localhost:11434"  # Not used for sentence-transformers, kept for compatibility


# 3. RAG configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 30
MAX_RETRIEVED_DOCS = 3


# 4. Amap API configuration
AMAP_API_KEY = "48257ed7b33d55e349260a9837436968" 

# 4. Database configuration
DB_PATH = r"D:\adavance\bigmodel\2.原创案例：Agentic RAG智能问答系统Agent\chinook.db"

# 5. LangChain configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# 6. Conversation history configuration
MAX_HISTORY_TURNS = 5

# 7. Document processing LLM configuration (for RAGAnything document processing)
# Used for entity extraction and summarization during document indexing
# Options: Use OpenAI models or Ollama models (if Ollama is running)
DOC_PROCESSING_LLM_MODEL = "gpt-4o-mini"  # Default: Use OpenAI (same as main agent)
# Alternative Ollama models (if you prefer local processing):
# DOC_PROCESSING_LLM_MODEL = "llama3.2"  # or "mistral", "qwen2.5", etc.
DOC_PROCESSING_USE_OPENAI = True  # Set to False to use Ollama for document processing 