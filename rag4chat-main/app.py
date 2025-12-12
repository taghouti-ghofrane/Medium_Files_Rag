# -*- coding: utf-8 -*-
# @Time    : 2025/08/06 10:00:00
# @Author  : nyzhhd
# @File    : app.py
# @Description: Main application file

# streamlit run app.py --server.port 6006

import streamlit as st

# At the very top of app.py (before imports) you can add:
# import os, tempfile
# # Point Streamlit temp to a directory you can read/write
# os.environ["STREAMLIT_TEMP_DIR"] = r"D:\streamlit_tmp"   # any valid path
# os.makedirs(os.environ["STREAMLIT_TEMP_DIR"], exist_ok=True)
# tempfile.tempdir = os.environ["STREAMLIT_TEMP_DIR"]

from datetime import datetime
import logging
import re
import asyncio
from config.settings import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    AVAILABLE_EMBEDDING_MODELS
)
# Assume these modules exist and are implemented
from models.agent import RAGAgent
from utils.chat_history import ChatHistoryManager
from utils.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from utils.ui_components import UIComponents
from utils.decorators import error_handler, log_execution
from utils.logger_config import get_logger

# Use centralized logging
logger = get_logger(__name__)


class App:
    """
    RAG application main class
    """

    def __init__(self):
        """
        @description Initialize application
        """
        logger.info("=" * 80)
        logger.info("Initializing RAG Application...")
        logger.info("=" * 80)
        
        self._init_session_state()  # initialize session state
        logger.debug("Session state initialized")
        
        self.chat_history = ChatHistoryManager()  # chat history manager
        logger.debug("Chat history manager initialized")
        
        self.document_processor = DocumentProcessor()  # document processor
        logger.debug("Document processor initialized")
        
        self.vector_store = VectorStoreService()  # vector store service
        logger.debug("Vector store service initialized")
        
        logger.info("✅ Application initialized successfully")

    # 1. Initialize session state
    @error_handler(show_error=False)
    def _init_session_state(self):
        """Ensure all session_state variables are initialized"""
        defaults = {
            'model_version': DEFAULT_MODEL,
            'processed_documents': [],
            'similarity_threshold': DEFAULT_SIMILARITY_THRESHOLD,
            'rag_enabled': True,
            'embedding_model': EMBEDDING_MODEL,
            'thinking': False,  # initialize thinking flag
            'user_input': "",   # initialize input box value
            '_input_to_process': "" # hold user input waiting to process
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # 2. Render sidebar
    @error_handler()
    @log_execution
    def render_sidebar(self):
        # Update model selection and embedding selection
        st.session_state.model_version, new_embedding_model = UIComponents.render_model_selection(
            AVAILABLE_MODELS,
            st.session_state.model_version,
            AVAILABLE_EMBEDDING_MODELS,
            st.session_state.embedding_model
        )

        # Check embedding model change
        previous_embedding_model = st.session_state.embedding_model
        st.session_state.embedding_model = new_embedding_model

        # Update RAG settings
        st.session_state.rag_enabled, st.session_state.similarity_threshold = UIComponents.render_rag_settings(
            st.session_state.rag_enabled,
            st.session_state.similarity_threshold,
            DEFAULT_SIMILARITY_THRESHOLD
        )

        # Update vector store embedding model
        if previous_embedding_model != st.session_state.embedding_model:
            if self.vector_store.update_embedding_model(st.session_state.embedding_model):
                # If vector store already exists, tell user to reprocess docs
                if len(st.session_state.processed_documents) > 0:
                    st.sidebar.info(
                    f"⚠️ Embedding model changed to {st.session_state.embedding_model}; you may need to reprocess documents.")

        # Render chat stats
        UIComponents.render_chat_stats(self.chat_history)

    # 3. Render document upload area
    @error_handler()
    @log_execution
    def render_document_upload(self):
        all_docs, self.vector_store = UIComponents.render_document_upload(
            self.document_processor,
            self.vector_store,
            st.session_state.processed_documents
        )

    # 4. Handle user input (start thinking workflow)
    @error_handler()
    @log_execution
    async def _handle_user_input_and_thinking(self, prompt: str):
        """
        Process user input and set thinking flag
        """
        self.chat_history.add_message("user", prompt)  # add user message
        st.session_state.thinking = True  # set thinking flag
        st.session_state._input_to_process = prompt # store pending input
        st.rerun()  # refresh UI to show thinking

    # 5. Process user input (actual handling)
    @error_handler()
    @log_execution
    async def process_user_input(self, prompt: str):
        """
        prompt - user prompt text

        1️⃣ RAG mode: search docs → build context → call model
        2️⃣ Plain mode: call model directly
        """
        logger.info("=" * 80)
        logger.info(f"Processing user input: '{prompt[:100]}...'")
        logger.info(f"RAG mode: {'ENABLED' if st.session_state.rag_enabled else 'DISABLED'}")
        
        # user message already added in _handle_user_input_and_thinking
        if st.session_state.rag_enabled:
            await self._process_rag_query(prompt)  # RAG path
        else:
            await self._process_simple_query(prompt)  # plain path

        # Clear flags after processing
        logger.debug("Clearing processing flags...")
        st.session_state.thinking = False
        st.session_state._input_to_process = ""
        logger.info("User input processing completed")
        st.rerun() # Refresh to show response

    # 6. Process RAG query
    @error_handler()
    @log_execution
    async def _process_rag_query(self, prompt: str):
        """
        prompt - user prompt text
        """
        logger.info(f"[RAG Query] Starting RAG query processing for: '{prompt[:50]}...'")
        top_k = 3
        logger.debug(f"[RAG Query] Using top_k={top_k}, similarity_threshold={st.session_state.similarity_threshold}")
        
        with st.spinner("🤔 Evaluating query..."):
            # Search related documents
            logger.info("[RAG Query] Step 1: Searching vector store for relevant documents...")
            docs = await self.vector_store.search_documents(
                prompt,
                top_k,
                st.session_state.similarity_threshold
            )
            logger.info(f"[RAG Query] Step 1 Complete: Retrieved {len(docs)} documents")
            if docs:
                logger.debug(f"[RAG Query] Top document preview: {docs[0].get('content', '')[:100]}...")
            
            # Build context
            logger.info("[RAG Query] Step 2: Building context from retrieved documents...")
            context = self.vector_store.get_context(docs)
            logger.debug(f"[RAG Query] Context length: {len(context)} characters")
            
            # Create RAG agent
            logger.info(f"[RAG Query] Step 3: Creating RAG agent with model: {st.session_state.model_version}")
            agent = RAGAgent(st.session_state.model_version)
            
            # Run agent to get response
            logger.info("[RAG Query] Step 4: Running agent to generate response...")
            response = agent.run(
                prompt,
                context=context
            )
            logger.info(f"[RAG Query] Step 4 Complete: Response generated ({len(response)} characters)")
            
            # Handle response
            logger.info("[RAG Query] Step 5: Processing and saving response...")
            await self._process_response(response, docs)
            logger.info("[RAG Query] ✅ RAG query processing completed successfully")

    # 7. Process simple query
    @error_handler()
    @log_execution
    async def _process_simple_query(self, prompt: str):
        """
        prompt - user prompt text
        """
        logger.info(f"[Simple Query] Starting simple query processing for: '{prompt[:50]}...'")
        logger.info(f"[Simple Query] Using model: {st.session_state.model_version}")
        
        with st.spinner("🤖 Thinking..."):
            # Create agent
            logger.debug("[Simple Query] Creating agent...")
            agent = RAGAgent(st.session_state.model_version)
            
            # Run agent
            logger.info("[Simple Query] Running agent to generate response...")
            response = agent.run(prompt)
            logger.info(f"[Simple Query] Response generated ({len(response)} characters)")
            
            # Handle response
            logger.debug("[Simple Query] Processing and saving response...")
            await self._process_response(response)
            logger.info("[Simple Query] ✅ Simple query processing completed successfully")

    # 8. Handle agent response
    async def _process_response(self, response: str, docs=None):
        """
        response - raw model response
        docs - retrieved documents (optional)
        """
        # 8.1 Extract think content
        think_pattern = r'<think>([\s\S]*?)</think>'  # pattern for think block
        think_match = re.search(think_pattern, response)
        if think_match:
            think_content = think_match.group(1).strip()
            response_wo_think = re.sub(think_pattern, '', response).strip()
        else:
            think_content = None
            response_wo_think = response

        # 8.2 Save response to history
        self.chat_history.add_message("assistant", response_wo_think)
        if think_content:
            self.chat_history.add_message("assistant_think", think_content)
        if docs:
            doc_contents = [doc.get('content', '') for doc in docs]  # avoid KeyError
            self.chat_history.add_message("retrieved_doc", doc_contents)

    # Entry point: run app
    @error_handler()
    @log_execution
    async def run(self):
        st.set_page_config(page_title="🐋 Your Smart Ops Assistant", layout="wide")
        UIComponents.inject_custom_css()  # inject custom CSS

        st.title("🐋 Your Smart Ops Assistant")
        st.info("🤖 Here to help you resolve issues quickly.")

        self.render_sidebar()
        self.render_document_upload()

        # --- Render chat history (above input box) ---
        UIComponents.render_chat_history(self.chat_history)

        # --- Render thinking hint ---
        if st.session_state.thinking:
             # Use separate container to avoid scroll issues
            with st.container():
                st.markdown('<div class="thinking-message">⏳ Thinking, please wait...</div>', unsafe_allow_html=True)
                UIComponents.scroll_to_bottom()

        # --- Render bottom input area ---
        # UIComponents handles the input interactions
        UIComponents.render_input_area()
        # Note: render_input_area uses callbacks; no direct return

        # --- Handle pending input logic ---
        # 1. If pending input and not thinking, start process
        pending_input = st.session_state.get('_input_to_process', '')
        if pending_input and not st.session_state.thinking:
            await self._handle_user_input_and_thinking(pending_input)

        # 2. If already thinking, continue processing
        elif st.session_state.thinking and pending_input:
             await self.process_user_input(pending_input)

        mode_description = ""
        if st.session_state.rag_enabled:
            mode_description += "📚 Ask about uploaded documents."
        else:
            mode_description += "💬 Chat directly with the model."
        mode_description += " 🔍 Handoff to human supported."
        mode_description += " 🌤️ Weather queries available."

        st.info(mode_description)


if __name__ == "__main__":
    app = App()  # Create application instance
    asyncio.run(app.run())  # Run application
