# -*- coding: utf-8 -*-
"""
Vector store service module (refactored version)
This version uses RAGDatabaseManager to manage documents and indexes,
and uses query_and_find_topk for queries.
"""
import os
import shutil
from typing import List, Optional, Dict, Any
import logging
import concurrent.futures
import functools
# Import new manager and query functions
# Assume these are in the same project or accessible via PYTHONPATH
# from .rag_database_manager import RAGDatabaseManager # If in package
# from .query_topk import query_and_find_topk # If in package
# Otherwise, adjust import path according to actual file structure, e.g.:
from services.build_database import RAGDatabaseManager  # Assume build_database.py is in same directory or imported
from services.get_top_from_rag import query_and_find_topk # Assume get_top_from_rag.py is in same directory or imported

# --- Configure logging ---
from utils.logger_config import get_logger
logger = get_logger(__name__)

class VectorStoreService:
    """
    Vector store service class for managing document vector storage (refactored version)
    This version delegates storage and indexing to RAGDatabaseManager, uses query_and_find_topk for queries.
    """

    # 1. Initialize vector store service
    def __init__(self, index_dir: str = "faiss_index", rag_working_dir: str = "./rag_storage_new"):
        """
        index_dir - Kept for compatibility, but mainly uses rag_working_dir
        rag_working_dir - RAG system working directory, contains vdb_chunks.json etc.
        """
        logger.info(f"Initializing VectorStoreService with working_dir: {rag_working_dir}")
        self.rag_working_dir = rag_working_dir
        self.index_dir = index_dir  # Kept for compatibility with old interface, but not actually used
        self.vdb_chunks_path = os.path.join(self.rag_working_dir, "vdb_chunks.json")
        logger.debug(f"Vector chunks path: {self.vdb_chunks_path}")

        # Initialize RAGDatabaseManager
        logger.debug("Initializing RAGDatabaseManager...")
        self.db_manager = RAGDatabaseManager(working_dir=self.rag_working_dir)
        # Add an attribute to indicate if service is ready (e.g., if index files exist)
        self.is_ready = False
        # Check once during initialization
        logger.debug("Loading vector store...")
        self.load_vector_store()
        logger.info(f"✅ VectorStoreService initialized - Ready: {self.is_ready}, Working dir: {self.rag_working_dir}")

    # 2. Update embedding model (no longer needed in this version, managed internally by RAGDatabaseManager)
    def update_embedding_model(self, model_name: str) -> bool:
        """
        This version does not directly support dynamic embedding model updates.
        Model configuration should be determined during RAGDatabaseManager initialization.
        @return Always returns False, indicating no update was performed.
        """
        logger.warning("update_embedding_model is not supported in this version. Configure model in RAGDatabaseManager.")
        return False

    # 3. Text chunking method (no longer needed in this version, handled internally by RAGDatabaseManager)
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        This version does not directly provide text chunking functionality.
        Chunking is completed internally by RAGDatabaseManager when processing documents.
        @return Directly returns original documents.
        """
        logger.warning("split_documents is handled internally by RAGDatabaseManager.")
        return documents

    # 4. Create new vector store instance (using RAGDatabaseManager)
    async def create_vector_store(self, document_paths: List[str]) -> bool:
        """
        Rebuild vector store using RAGDatabaseManager.
        document_paths - List of local document file paths (PDF, DOCX, etc.)
        @return Whether creation/rebuild was successful
        """
        if not document_paths:
            logger.warning("No document paths to create vector store")
            # Update status
            self.is_ready = False
            return False

        logger.info(f"Starting to rebuild vector store via RAGDatabaseManager, document count: {len(document_paths)}")

        try:
            # Rebuild database using RAGDatabaseManager
            # This clears existing data and adds new documents
            success = await self.db_manager.add_documents(document_paths)

            if success:
                logger.info("Vector store (via RAGDatabaseManager) rebuild successful")
                # Update status
                self.is_ready = True
            else:
                logger.error("Vector store (via RAGDatabaseManager) rebuild failed")
                # Update status
                self.is_ready = False
            return success

        except Exception as e:
            logger.error(f"Failed to create/rebuild vector store via RAGDatabaseManager: {str(e)}", exc_info=True)
            # Update status
            self.is_ready = False
            return False

    # 5. Save vector store (this functionality is automatically handled internally by RAGDatabaseManager)
    def _save_vector_store(self):
        """
        In this version, saving is automatically handled by RAGDatabaseManager.
        """
        logger.info("Vector store saving is automatically handled by RAGDatabaseManager.")

    # 6. Load vector store (check file existence and set is_ready)
    def load_vector_store(self) -> bool:
        """
        Check if vector store files exist to determine if loaded or loadable.
        @return bool: Returns True if vector store files are detected, otherwise False.
        """
        # Check if core vdb_chunks.json file exists
        if os.path.exists(self.vdb_chunks_path):
            logger.info(f"Detected vector store file: {self.vdb_chunks_path}")
            # Update status
            self.is_ready = True
            return True  # Return True to indicate ready
        else:
            logger.warning(f"Vector store file not detected: {self.vdb_chunks_path}")
            # Update status
            self.is_ready = False
            return False  # Return False to indicate not ready

    # 7. Search related documents (core functionality, uses new query_and_find_topk)
    async def search_documents(self, query: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        query - Query text
        top_k - Top K results to return
        threshold - Similarity threshold (may need adjustment in query_and_find_topk)

        @return List of related document information, each element is a dict containing 'content', 'file_path', 'similarity' etc.
                Returns empty list if error occurs or nothing found.
        """
        logger.info(f"[Vector Search] Starting document search - Query: '{query[:50]}...', top_k: {top_k}, threshold: {threshold}")
        
        # Check if vector store is ready
        if not self.is_ready:
            logger.warning("[Vector Search] Vector store not ready, cannot perform search")
            return []

        # Check if vector store file exists (double check)
        if not os.path.exists(self.vdb_chunks_path):
            logger.warning(f"[Vector Search] Vector store file does not exist: {self.vdb_chunks_path}")
            # Status may be out of sync, update it
            self.is_ready = False
            return []

        try:
            logger.debug(f"[Vector Search] Calling query_and_find_topk with query, path: {self.vdb_chunks_path}, top_k: {top_k}")
            
            # --- Key: Call new query function ---
            # Note: This requires query_and_find_topk to be modified to return structured data
            # rather than just printing results.
            # If query_and_find_topk still only prints, refer to previous answer to modify it.
            results = await query_and_find_topk(query, self.vdb_chunks_path, top_k)
            logger.debug(f"[Vector Search] query_and_find_topk returned {len(results) if isinstance(results, list) else 'non-list'} results")

            # Ensure return is a list
            if not isinstance(results, list):
                 logger.error(f"[Vector Search] query_and_find_topk returned unexpected type: {type(results)}")
                 return []

            # Filter results by threshold (if needed)
            # Note: query_and_find_topk internally calculates cosine similarity, range [0, 1] (usually)
            # threshold=0.0 means return all results
            if threshold > 0.0:
                filtered_results = [res for res in results if res.get('similarity', 1.0) >= threshold]
                logger.info(f"[Vector Search] ✅ Search completed - {len(filtered_results)}/{len(results)} documents passed threshold ({threshold})")
                if filtered_results:
                    logger.debug(f"[Vector Search] Top result similarity: {filtered_results[0].get('similarity', 'N/A')}")
                return filtered_results
            else:
                logger.info(f"[Vector Search] ✅ Search completed - Returning {len(results)} related documents")
                if results:
                    logger.debug(f"[Vector Search] Top result similarity: {results[0].get('similarity', 'N/A')}")
                return results

        except Exception as e:
            logger.error(f"[Vector Search] ❌ Failed to search documents: {str(e)}", exc_info=True)
            return []

    # 8. Get document context (unchanged, processes search_documents output)
    def get_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        docs - List of document dictionaries returned by search_documents

        @return Merged context
        """
        if not docs:
            return ""
        # Assume each doc dict has 'content' key
        return "\n\n".join(doc.get('content', '') for doc in docs)

    # 9. Add single document to vector store (using RAGDatabaseManager)
    async def add_document(self, document_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add single document to vector store (via RAGDatabaseManager)

        @param {str} document_path - Document file path (PDF, DOCX, etc.)
        @param {Dict[str, Any]} metadata - Document metadata (optional, handled by RAGDatabaseManager)
        @return {bool} Whether addition was successful
        """
        logger.info(f"[Add Document] Starting to add document: {document_path}")
        
        if not document_path or not os.path.exists(document_path):
            logger.warning(f"[Add Document] Document path invalid or file does not exist: '{document_path}'")
            return False

        try:
            file_size = os.path.getsize(document_path)
            logger.debug(f"[Add Document] File size: {file_size} bytes")

            # Use RAGDatabaseManager to add single document
            # Note: Ensure RAGDatabaseManager.add_document returns boolean
            logger.info(f"[Add Document] Calling RAGDatabaseManager.add_document...")
            success = await self.db_manager.add_document(document_path)

            if success:
                logger.info(f"[Add Document] ✅ Successfully added document: {document_path}")
                # After adding document, index should be updated, mark as ready
                self.is_ready = True
                logger.debug(f"[Add Document] Vector store marked as ready")
            else:
                logger.error(f"[Add Document] ❌ Failed to add document: {document_path}")
            return success

        except Exception as e:
            logger.error(f"[Add Document] ❌ Exception while adding document '{document_path}': {str(e)}", exc_info=True)
            return False

    # 10. Clear index (using RAGDatabaseManager logic or direct file operations)
    def clear_index(self):
        """
        Clear index (via RAGDatabaseManager or direct file operations)
        """
        try:
            # Method 1: Directly delete related files under rag_working_dir
            # This is the most direct and thorough method
            if os.path.exists(self.rag_working_dir):
                shutil.rmtree(self.rag_working_dir)
                logger.info(f"Index directory '{self.rag_working_dir}' cleared")
                # Recreate empty directory
                os.makedirs(self.rag_working_dir, exist_ok=True)
                # Reinitialize db_manager instance
                self.db_manager = RAGDatabaseManager(working_dir=self.rag_working_dir)
                logger.info("RAGDatabaseManager reinitialized")
            else:
                logger.warning(f"Index directory '{self.rag_working_dir}' does not exist")

            # After clearing, status becomes not ready
            self.is_ready = False

        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}", exc_info=True)
            # Even if error occurs, may consider index status uncertain, set to False
            self.is_ready = False
            raise # Re-raise exception for caller to handle

    # --- Compatibility property (for old UI checks) ---
    @property
    def vector_store(self):
        """
        Compatibility property: Provides checkpoint for old UI.
        If service is ready (is_ready=True), returns a non-None placeholder object.
        Otherwise returns None.
        """
        # Note: This placeholder object should not have its methods actually called
        # It is merely to make checks like `if not vector_store.vector_store:` pass or fail
        if self.is_ready:
            # Return a simple non-None object as placeholder
            return lambda: None # or object() or type('Placeholder', (), {})()
        else:
            return None

