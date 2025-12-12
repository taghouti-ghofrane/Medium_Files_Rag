"""
Document processing module
"""
import os
import hashlib
from pathlib import Path
# Can live outside or inside __init__
STATIC_DIR = Path("static").resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
import json
from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path
import io
import tempfile
from utils.decorators import error_handler, log_execution
from datetime import datetime
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
from utils.logger_config import get_logger
logger = get_logger(__name__)

class DocumentProcessor:
    """
    Document processor for PDFs and other files
    """

    # 1. Initialize document processor
    def __init__(self, cache_dir: str = ".cache", max_workers: int = 4):
        """
        cache_dir - cache directory
        max_workers - max workers
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
            is_separator_regex=False
        )
    
    # 2. Get cache file path
    def _get_cache_path(self, file_content: bytes, file_name: str) -> Path:
        """
        file_content - file bytes
        file_name - file name

        @return cache path
        """
        cache_key = hashlib.md5(file_content + file_name.encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.json"
    
    # 3. Load processing results from cache
    def _load_from_cache(self, cache_path: str) -> Optional[List[Document]]:
        """
        cache_path - cache file
        return cached docs or None
        """
        try:
            path = Path(cache_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [Document(**doc) for doc in data]
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
        return None
    
    # 4. Save processing results to cache
    def _save_to_cache(self, cache_path: Path, documents: List[Document]):
        """
        cache_path - cache file
        documents - processed docs
        """
        try:
            # Convert Document objects to serializable dictionaries
            docs_data = [doc.dict() for doc in documents]
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    

    # 5. Process PDF file
    @error_handler()
    @log_execution
    def _process_pdf(self, file_content: bytes, file_name: str) -> List[Document]:
        """
        file_content - PDF bytes
        file_name - PDF file name

        @return processed documents
        """
        logger.info(f"[Document Processor] Processing PDF: {file_name} ({len(file_content)} bytes)")
        
        # Check cache
        cache_path = self._get_cache_path(file_content, file_name)
        logger.debug(f"[Document Processor] Cache path: {cache_path}")
        cached_docs = self._load_from_cache(str(cache_path))
        if cached_docs is not None:
            logger.info(f"[Document Processor] ✅ Loading file from cache: {file_name} ({len(cached_docs)} chunks)")
            return cached_docs
        
        # Process PDF
        logger.info(f"[Document Processor] Processing file (not in cache): {file_name}")
        
        try:
            # Create temporary file, use context manager for automatic cleanup
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                # Load PDF from temp file
                logger.debug(f"[Document Processor] Loading PDF with PyPDFLoader...")
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()
                logger.info(f"[Document Processor] PDF loaded: {len(documents)} pages")
                
                # Split documents
                logger.debug(f"[Document Processor] Splitting documents into chunks...")
                split_docs = self.text_splitter.split_documents(documents)
                logger.info(f"[Document Processor] Documents split into {len(split_docs)} chunks")
                
                # Save cache
                if split_docs:
                    logger.debug(f"[Document Processor] Saving to cache...")
                    self._save_to_cache(cache_path, split_docs)
                    logger.info(f"[Document Processor] ✅ Cache saved successfully")
                
                return split_docs
                
        except Exception as e:
            logger.error(f"Failed to process PDF file: {str(e)}")
            raise
    

    # 6. Clear all cache
    def clear_cache(self):
        try:
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise


    # 7. Process uploaded file, supports multiple file types
    @error_handler()
    @log_execution
    def process_file(self, uploaded_file_or_content, file_name: str = None) -> Union[str, List[Document]]:
        """
        Unified entry: PDF/TXT/DOCX/MD supported.
        Returns:
            - Streamlit upload → full text (str)
            - bytes + file_name → List[Document]
        """
        logger.info(f"[Document Processor] Processing file: {file_name}")
        try:
            # 1. Normalize file_content and file_name
            if hasattr(uploaded_file_or_content, 'getvalue') and hasattr(uploaded_file_or_content, 'name'):
                file_content = uploaded_file_or_content.getvalue()
                file_name = uploaded_file_or_content.name
                logger.debug(f"[Document Processor] File from Streamlit upload: {file_name} ({len(file_content)} bytes)")
            elif isinstance(uploaded_file_or_content, bytes) and file_name:
                file_content = uploaded_file_or_content
                logger.debug(f"[Document Processor] File from bytes: {file_name} ({len(file_content)} bytes)")
            else:
                raise ValueError("Invalid parameters: need file object or bytes with file_name")

            # 2. Ensure safe directory exists (keep within repo to avoid drive issues)
            safe_dir = Path("static/uploads")
            safe_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[Document Processor] Safe directory: {safe_dir}")

            # 3. Copy file to safe dir using only the filename (strip any path)
            safe_path = safe_dir / Path(file_name).name
            logger.debug(f"[Document Processor] Saving file to: {safe_path}")
            with open(safe_path, "wb") as f_out:
                f_out.write(file_content)
            logger.info(f"[Document Processor] ✅ File saved to safe directory")

            # 4. Process based on suffix
            file_ext = file_name.lower().split('.')[-1] if '.' in file_name else 'unknown'
            logger.info(f"[Document Processor] Processing {file_ext.upper()} file...")
            
            if file_name.lower().endswith('.pdf'):
                logger.debug(f"[Document Processor] Processing PDF file...")
                docs = self._process_pdf_from_path(safe_path)
                logger.info(f"[Document Processor] ✅ PDF processed: {len(docs)} documents")
                return "\n\n".join(doc.page_content for doc in docs) \
                       if hasattr(uploaded_file_or_content, 'getvalue') else docs

            elif file_name.lower().endswith('.txt'):
                logger.debug(f"[Document Processor] Processing TXT file...")
                # Return a Document with the stored path so downstream indexing can read the file
                txt = safe_path.read_text(encoding='utf-8')
                logger.info(f"[Document Processor] ✅ TXT processed: {len(txt)} characters")
                return Document(
                    page_content=txt,
                    metadata={"source": str(safe_path)}
                )

            else:
                logger.warning(f"[Document Processor] ⚠️ Unsupported file type: {file_name}")
                return f"Unsupported file type: {file_name}"

        except Exception as e:
            logger.error(f"[Document Processor] ❌ Failed to process file: {str(e)}", exc_info=True)
            raise Exception(f"Failed to process file: {str(e)}")
    @error_handler()
    @log_execution
    def _process_pdf_from_path(self, pdf_path: Path) -> List[Document]:
        """
        Read local PDF and return Document list
        """
        logger.info(f"[Document Processor] Processing PDF from path: {pdf_path}")
        import fitz  # PyMuPDF
        docs = []
        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
            logger.debug(f"[Document Processor] PDF has {page_count} pages")
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                logger.debug(f"[Document Processor] Page {page_num}: {len(text)} characters")
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "file_path": str(pdf_path)}
                    )
                )
        logger.info(f"[Document Processor] ✅ PDF processed: {len(docs)} pages extracted")
        return docs