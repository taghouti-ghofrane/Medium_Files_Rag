import asyncio
import aiohttp
import base64
from typing import List, Dict, Any, Optional
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import os
import shutil
import logging
import numpy as np
import json


from utils.logger_config import get_logger
from config.settings import (
    DOC_PROCESSING_LLM_MODEL,
    DOC_PROCESSING_USE_OPENAI,
    OPENAI_API_KEY,
    OPENAI_BASE_URL
)
logger = get_logger(__name__)

# =======================
# Removed AsyncOpenAI import - using sentence-transformers instead (open source)
# =======================



# If you want to use rerank, install sentence-transformers
try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False


# =======================
# Utility function: Sentence Transformers Embedding (Open Source)
# =======================
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Global model cache to avoid reloading
_embedding_model_cache = {}

async def sentence_transformers_embed_async(texts: List[str], model: str = "BAAI/bge-large-en-v1.5") -> List[List[float]]:
    """
    Async call to sentence-transformers for embeddings (open source, no API key needed)
    
    Args:
        texts: List of text strings to embed
        model: Model name from Hugging Face (default: BAAI/bge-large-en-v1.5 - 1024 dim)
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        error_msg = (
            "❌ sentence-transformers is not installed!\n"
            "📦 Install with: pip install sentence-transformers\n"
            "💡 This is required for document embedding. Without it, chunks cannot be created."
        )
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    # Load model (cache it to avoid reloading)
    if model not in _embedding_model_cache:
        logger.info(f"Loading sentence-transformers model: {model}")
        _embedding_model_cache[model] = SentenceTransformer(model)
        logger.info(f"✅ Model loaded successfully")
    
    model_instance = _embedding_model_cache[model]
    target_dim = model_instance.get_sentence_embedding_dimension()
    
    embeddings = []
    
    # Preprocess texts
    processed_texts = []
    for i, text_item in enumerate(texts):
        # Type check and preprocessing
        if not isinstance(text_item, str):
            if text_item is None:
                logger.warning(f"sentence_transformers_embed_async: Text {i} is None, converting to empty string.")
                text = ""
            else:
                logger.warning(f"sentence_transformers_embed_async: Text {i} type is {type(text_item)}, attempting to convert to string.")
                text = str(text_item)
        else:
            text = text_item

        # Handle empty or whitespace-only strings
        if not text or not text.strip():
            logger.debug(f"sentence_transformers_embed_async: Text {i} is empty, returning zero vector.")
            embeddings.append([0.0] * target_dim)
            continue
        
        processed_texts.append((i, text))
    
    if not processed_texts:
        return embeddings
    
    # Generate embeddings in batch (sentence-transformers handles this efficiently)
    try:
        # Extract just the texts for batch processing
        texts_to_embed = [text for _, text in processed_texts]
        indices = [idx for idx, _ in processed_texts]
        
        # Run in executor to make it async-friendly (sentence-transformers is synchronous)
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Encode in batch (much faster than one-by-one)
        embedding_vectors = await loop.run_in_executor(
            None,
            lambda: model_instance.encode(
                texts_to_embed,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )
        )
        
        # Map embeddings back to original indices
        embedding_dict = {idx: embedding_vectors[i].tolist() for i, idx in enumerate(indices)}
        
        # Reconstruct in original order (including empty texts that got zero vectors)
        for i in range(len(texts)):
            if i in embedding_dict:
                embedding = embedding_dict[i]
                # Validate embedding
                import math
                if any(math.isnan(x) or math.isinf(x) for x in embedding):
                    logger.error(f"sentence_transformers_embed_async: Embedding vector contains NaN or Inf (text {i})")
                    embeddings.append([0.0] * target_dim)
                else:
                    embeddings.append(embedding)
                    logger.debug(f"sentence_transformers_embed_async: Successfully generated {len(embedding)}-dimensional embedding for text {i}.")
            # else: already added zero vector above
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings using {model} (dimension: {target_dim})")
        
    except Exception as e:
        logger.error(f"Error occurred when generating embeddings with sentence-transformers: {e}", exc_info=True)
        # Return zero vectors for failed texts
        for i in range(len(texts) - len(embeddings)):
            embeddings.append([0.0] * target_dim)

    return embeddings

async def ollama_complete_async(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    **kwargs
) -> str:
    """
    Async call to Ollama (for text LLM), supports text input
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                error_msg = f"Ollama LLM error {resp.status}: {text}"
                logger.error(f"[Ollama LLM] {error_msg}")
                # Try to parse error message
                try:
                    error_json = await resp.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        error_detail = error_json["error"]["message"]
                        if "not found" in error_detail.lower():
                            logger.error(f"[Ollama LLM] Model '{model}' not found. Available models might be different.")
                            logger.error(f"[Ollama LLM] 💡 Try: ollama pull {model} or use a different model")
                except:
                    pass
                raise Exception(error_msg)
            result = await resp.json()
            return result["choices"][0]["message"]["content"]


async def ollama_vision_complete_async(
    model: str,
    prompt: str,
    image_data: str,  # Base64 encoded image string
    **kwargs
) -> str:
    """
    Async call to Ollama vision model (e.g., Llava), supports image + text input
    Uses /api/generate endpoint
    """
    # Build payload conforming to Ollama vision model requirements
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],  # Image data passed as list
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/api/generate", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                error_msg = f"Ollama Vision Model error {resp.status}: {text}"
                logger.error(f"[Ollama Vision] {error_msg}")
                try:
                    error_json = await resp.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        error_detail = error_json["error"]["message"]
                        if "not found" in error_detail.lower():
                            logger.error(f"[Ollama Vision] Model '{model}' not found.")
                            logger.error(f"[Ollama Vision] 💡 Try: ollama pull {model} or use a different model")
                except:
                    pass
                raise Exception(error_msg)
            result = await resp.json()
            # Vision model response is directly in 'response' field
            return result.get("response", "")


# =======================
# Async embedding function wrapper (using sentence-transformers)
# =======================
class AsyncEmbeddingWrapper:
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize embedding wrapper with sentence-transformers model
        
        Args:
            model: Hugging Face model name
                - "BAAI/bge-large-en-v1.5" (1024 dim) - Best quality, recommended
                - "BAAI/bge-base-en-v1.5" (768 dim) - Good balance
                - "all-mpnet-base-v2" (768 dim) - General purpose
        """
        self.model = model
        logger.info(f"Initialized AsyncEmbeddingWrapper with model: {model}")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await sentence_transformers_embed_async(texts, self.model)


# =======================
# Rerank function (optional)
# =======================

rerank_model_func = None
if RERANK_AVAILABLE:
    try:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")

        def rerank_model_func(query: str, docs: List[str]) -> List[str]:
            if not docs:
                return []
            pairs = [(query, doc) for doc in docs]
            scores = _reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked]
    except Exception as e:
        logger.warning(f"Rerank model initialization failed: {e}")
        rerank_model_func = None


# =======================
# Database construction and update function interface
# =======================

class RAGDatabaseManager:
    """RAG Database Manager - Specifically responsible for database construction and updates"""

    def __init__(
        self,
        working_dir: str = "./rag_storage_new",
        output_dir: str = "./output",
        llm_model: str = None,  # None = use default from settings
        embed_model: str = "BAAI/bge-large-en-v1.5",
        vision_model: str = "llava:latest",
        parser: str = "mineru"
    ):
        """
        Initialize database manager
        Args:
            working_dir: RAG storage directory
            output_dir: Output directory
            llm_model: LLM model name
            embed_model: Embedding model name
            vision_model: Vision model name
            parser: Document parser
        """
        self.working_dir = working_dir
        self.output_dir = output_dir
        # Use default from settings if not provided
        self.llm_model = llm_model if llm_model is not None else DOC_PROCESSING_LLM_MODEL
        self.embed_model = embed_model
        self.vision_model = vision_model
        self.parser = parser
        self.use_openai_for_llm = DOC_PROCESSING_USE_OPENAI
        
        logger.info(f"[RAGDatabaseManager] Initialized with LLM: {self.llm_model} (OpenAI: {self.use_openai_for_llm})")
        
        # Verify sentence-transformers is available at initialization
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error(
                "❌ sentence-transformers is not installed!\n"
                "📦 Install with: pip install sentence-transformers\n"
                "💡 Without embeddings, chunks cannot be created."
            )
            # Don't raise here, let it fail later with a clearer message
        
        # Create embedding wrapper instance
        try:
            self.embedding_wrapper = AsyncEmbeddingWrapper(self.embed_model)
            logger.info(f"✅ Embedding wrapper initialized with model: {self.embed_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding wrapper: {e}")
            raise
        
        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"Working directory: {self.working_dir}, Output directory: {self.output_dir}")

    async def _create_rag_instance(self):
        """Create RAG instance"""
        
        # Verify OpenAI configuration if using OpenAI
        if self.use_openai_for_llm:
            if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
                error_msg = (
                    f"❌ Invalid or missing OPENAI_API_KEY!\n"
                    f"Current value: {'Set' if OPENAI_API_KEY else 'Not set'}\n"
                    f"💡 Solutions:\n"
                    f"  1. Set OPENAI_API_KEY in .env file\n"
                    f"  2. Or set DOC_PROCESSING_USE_OPENAI=False to use Ollama instead"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Create configuration
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser=self.parser,
            parse_method="auto",
            enable_image_processing=True, # Keep image processing enabled
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # LLM function (async) - for text processing
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # Force OpenAI if configured (prevent accidental Ollama usage)
            if not self.use_openai_for_llm:
                logger.warning(f"[LLM Func] ⚠️ WARNING: use_openai_for_llm=False, will use Ollama model: {self.llm_model}")
            else:
                logger.info(f"[LLM Func] ✅ Using OpenAI model: {self.llm_model}")
            
            if self.use_openai_for_llm:
                # Use OpenAI for document processing
                try:
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(
                        api_key=OPENAI_API_KEY,
                        base_url=OPENAI_BASE_URL
                    )
                    
                    # Build messages
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    for msg in history_messages:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    messages.append({"role": "user", "content": prompt})
                    
                    logger.debug(f"[OpenAI LLM] Calling OpenAI model: {self.llm_model}")
                    # Filter out unsupported kwargs (lightrag passes 'hashing_kv' which OpenAI doesn't support)
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['hashing_kv']}
                    # Call OpenAI API
                    response = await client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        **filtered_kwargs
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"[OpenAI LLM] ❌ Error calling OpenAI API: {e}")
                    error_msg = (
                        f"Failed to call OpenAI API for document processing.\n"
                        f"Model: {self.llm_model}\n"
                        f"Error: {str(e)}\n"
                        f"💡 Solutions:\n"
                        f"  1. Check your OPENAI_API_KEY in .env file\n"
                        f"  2. Verify your OpenAI API key is valid and has credits\n"
                        f"  3. Check your internet connection\n"
                        f"  4. Verify the model name '{self.llm_model}' is correct\n"
                        f"  5. Or set DOC_PROCESSING_USE_OPENAI=False in config/settings.py to use Ollama instead"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg) from e
            else:
                # Use Ollama for document processing
                logger.debug(f"[Ollama LLM] Calling Ollama model: {self.llm_model}")
                try:
                    return await ollama_complete_async(
                        model=self.llm_model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"[Ollama LLM] ❌ Error calling Ollama API: {e}")
                    error_msg = (
                        f"Failed to call Ollama API for document processing.\n"
                        f"Model: {self.llm_model}\n"
                        f"Error: {str(e)}\n"
                        f"💡 Solutions:\n"
                        f"  1. Make sure Ollama is running: ollama serve\n"
                        f"  2. Pull the model: ollama pull {self.llm_model}\n"
                        f"  3. Check available models: ollama list\n"
                        f"  4. Or set DOC_PROCESSING_USE_OPENAI=True in config/settings.py to use OpenAI instead"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg) from e

        # Vision model function (async) - for image processing, uses new vision model API
        async def vision_model_func(
            prompt,
            system_prompt=None, # Note: Ollava's /api/generate may not directly support system prompt
            history_messages=[], # Note: Ollava's /api/generate may not directly support history messages
            image_data=None,
            **kwargs
        ):
            # If image data is provided, call vision model
            if image_data:
                # Can combine prompt and system_prompt here if needed
                # But standard /api/generate interface may not distinguish them
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"

                return await ollama_vision_complete_async(
                    model=self.vision_model,
                    prompt=full_prompt,
                    image_data=image_data,
                    **kwargs
                )
            else:
                # If no image data, fallback to regular LLM
                return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Embedding function (truly async function)
        async def async_embedding_func(texts: List[str]) -> List[List[float]]:
            # Filter empty texts
            filtered_texts = [text if text.strip() else " " for text in texts]
            return await self.embedding_wrapper.embed(filtered_texts)

        # Get actual embedding dimension from the model
        # For bge-large-en-v1.5 it's 1024, for bge-base it's 768
        # We'll detect it dynamically
        try:
            if "bge-large" in self.embed_model.lower():
                embedding_dim = 1024
            elif "bge-base" in self.embed_model.lower():
                embedding_dim = 768
            elif "mpnet" in self.embed_model.lower():
                embedding_dim = 768
            else:
                # Default to 1024, will be adjusted if needed
                embedding_dim = 1024
                logger.warning(f"Unknown model {self.embed_model}, defaulting to 1024 dimensions")
        except:
            embedding_dim = 1024
        
        # Wrap embedding function in EmbeddingFunc format
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,  # Dynamic dimension based on model
            max_token_size=512,  # Support long text
            func=async_embedding_func,
        )
        logger.info(f"Embedding function configured with dimension: {embedding_dim}")
        # === Add these three lines to save functions as instance attributes ===
        self.llm_model_func = llm_model_func
        self.vision_model_func = vision_model_func
        self.embedding_func = embedding_func

        # Initialize RAGAnything parameters
        rag_args = {
            "config": config,
            "llm_model_func": llm_model_func,
            "vision_model_func": vision_model_func, # Use modified vision model function
            "embedding_func": embedding_func,
        }

        # Only add rerank_model_func if supported
        if rerank_model_func is not None:
            try:
                rag = RAGAnything(**rag_args, rerank_model_func=rerank_model_func)
            except TypeError:
                logger.warning("Current RAGAnything version doesn't support rerank_model_func, initializing without it")
                rag = RAGAnything(**rag_args)
        else:
            rag = RAGAnything(**rag_args)
        return rag

    async def add_document(
        self,
        file_path: str,
        parse_method: str = "auto"
    ) -> bool:
        """
        Add a single document to the database
        Args:
            file_path: Document path
            parse_method: Parse method
        Returns:
            bool: Whether addition was successful
        """
        logger.info("=" * 80)
        logger.info(f"[Add Document] Starting document addition process")
        logger.info(f"[Add Document] File: {file_path}")
        logger.info(f"[Add Document] Parse method: {parse_method}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"[Add Document] ❌ File does not exist: {file_path}")
            return False
        
        # Get file info
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.debug(f"[Add Document] File size: {file_size} bytes, Extension: {file_extension}")
        
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
        if file_extension not in supported_extensions:
            logger.warning(f"[Add Document] ⚠️ File format may not be supported: {file_extension}, will attempt to process")
        
        try:
            # Create RAG instance
            logger.info(f"[Add Document] Step 1: Creating RAG instance...")
            rag = await self._create_rag_instance()
            logger.debug(f"[Add Document] RAG instance created successfully")
            
            # Process document
            logger.info(f"[Add Document] Step 2: Processing document '{file_path}'...")
            logger.info(f"[Add Document] 🚀 Starting document processing...")
            await rag.process_document_complete(
                file_path=file_path,
                output_dir=self.output_dir,
                parse_method=parse_method
            )
            logger.info(f"[Add Document] ✅ Document processing completed successfully: {file_path}")
            logger.info("=" * 80)
            return True
        except Exception as e:
            logger.error(f"[Add Document] ❌ Document processing failed: {file_path}")
            logger.error(f"[Add Document] Error details: {str(e)}", exc_info=True)
            import traceback
            traceback.print_exc()
            logger.info("=" * 80)
            return False

    async def add_documents(
        self,
        file_paths: List[str],
        parse_method: str = "auto"
    ) -> Dict[str, bool]:
        """
        Batch add multiple documents to the database
        Args:
            file_paths: List of document paths
            parse_method: Parse method
        Returns:
            Dict[str, bool]: Processing result for each document
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = await self.add_document(file_path, parse_method)
        return results

    async def rebuild_database(
        self,
        file_paths: List[str],
        clear_existing: bool = True
    ) -> bool:
        """
        Rebuild database (optionally clear existing data)
        Args:
            file_paths: List of document paths to add
            clear_existing: Whether to clear existing data
        Returns:
            bool: Whether rebuild was successful
        """
        try:
            # If need to clear existing data
            if clear_existing and os.path.exists(self.working_dir):
                import shutil
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
                logger.info("🧹 Cleared existing database")
            # Batch add documents
            results = await self.add_documents(file_paths)
            # Check results
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            logger.info(f"📊 Database rebuild completed: {success_count}/{total_count} documents processed successfully")
            return success_count == total_count
        except Exception as e:
            logger.error(f"❌ Database rebuild failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information
        Returns:
            Dict: Database information
        """
        info = {
            "working_dir": self.working_dir,
            "output_dir": self.output_dir,
            "models": {
                "llm": self.llm_model,
                "embedding": self.embed_model,
                "vision": self.vision_model
            }
        }
        # Count storage files
        if os.path.exists(self.working_dir):
            files = os.listdir(self.working_dir)
            info["storage_files"] = files
            info["document_count"] = len([f for f in files if f.endswith('.json')])
        else:
            info["storage_files"] = []
            info["document_count"] = 0
        return info


# =======================
# Convenience function interface
# =======================

async def add_document_to_rag(
    file_path: str,
    working_dir: str = "./rag_storage_new",
    output_dir: str = "./output",
    **kwargs
) -> bool:
    """
    Add a single document to RAG database (async interface)
    Args:
        file_path: Document path
        working_dir: Database storage directory
        output_dir: Output directory
        **kwargs: Other parameters (llm_model, embed_model, vision_model, parser)
    Returns:
        bool: Whether addition was successful
    """
    manager = RAGDatabaseManager(
        working_dir=working_dir,
        output_dir=output_dir,
        **kwargs
    )
    return await manager.add_document(file_path)


async def add_documents_to_rag(
    file_paths: List[str],
    working_dir: str = "./rag_storage_new",
    output_dir: str = "./output",
    **kwargs
) -> Dict[str, bool]:
    """
    Batch add documents to RAG database (async interface)
    Args:
        file_paths: List of document paths
        working_dir: Database storage directory
        output_dir: Output directory
        **kwargs: Other parameters
    Returns:
        Dict[str, bool]: Processing result for each document
    """
    manager = RAGDatabaseManager(
        working_dir=working_dir,
        output_dir=output_dir,
        **kwargs
    )
    return await manager.add_documents(file_paths)


async def rebuild_rag_database(
    file_paths: List[str],
    working_dir: str = "./rag_storage_new",
    output_dir: str = "./output",
    clear_existing: bool = True,
    **kwargs
) -> bool:
    """
    Rebuild RAG database (async interface)
    Args:
        file_paths: List of document paths
        working_dir: Database storage directory
        output_dir: Output directory
        clear_existing: Whether to clear existing data
        **kwargs: Other parameters
    Returns:
        bool: Whether rebuild was successful
    """
    manager = RAGDatabaseManager(
        working_dir=working_dir,
        output_dir=output_dir,
        **kwargs
    )
    return await manager.rebuild_database(file_paths, clear_existing)


# =======================
# Synchronous interface
# =======================

def add_document_sync(
    file_path: str,
    working_dir: str = "./rag_storage_new",
    output_dir: str = "./output",
    **kwargs
) -> bool:
    """Synchronous interface: Add single document"""
    return asyncio.run(add_document_to_rag(file_path, working_dir, output_dir, **kwargs))


def add_documents_sync(
    file_paths: List[str],
    working_dir: str = "./rag_storage_new",
    output_dir: str = "./output",
    **kwargs
) -> Dict[str, bool]:
    """Synchronous interface: Batch add documents"""
    return asyncio.run(add_documents_to_rag(file_paths, working_dir, output_dir, **kwargs))


def rebuild_database_sync(
    file_paths: List[str],
    working_dir: str = "./rag_storage_new",
    output_dir: str = "./output",
    clear_existing: bool = True,
    **kwargs
) -> bool:
    """Synchronous interface: Rebuild database"""
    return asyncio.run(rebuild_rag_database(file_paths, working_dir, output_dir, clear_existing, **kwargs))


# =======================
# Usage example
# =======================

async def main():
    # Create database manager
    db_manager = RAGDatabaseManager(
        working_dir="./rag_storage_new",
        output_dir="./output"
    )
    # Example 1: Add single document
    # pdf_path = r"D:\adavance\tsy\rag4chat\test.pdf"
    # success = await db_manager.add_document(pdf_path)
    # # print(f"Add document result: {success}")

    # Example 2: Batch add documents
    documents = [
        r"D:\adavance\tsy\rag4chat\knowladge2.docx",
        r"D:\adavance\tsy\rag4chat\knowladge3.docx",
        r"D:\adavance\tsy\rag4chat\knowladge4.docx",
        # r"D:\adavance\tsy\rag4chat\test.md" # If there are other documents
    ]
    results = await db_manager.add_documents(documents)
    print(f"Batch add results: {results}")

    # # # Example 3: Get database information
    # info = db_manager.get_database_info()
    # print(f"Database information: {info}")

    # # Example 4: Rebuild database
    # rebuild_success = await db_manager.rebuild_database(documents, clear_existing=True)
    # print(f"Rebuild database result: {rebuild_success}")


if __name__ == "__main__":
    asyncio.run(main())