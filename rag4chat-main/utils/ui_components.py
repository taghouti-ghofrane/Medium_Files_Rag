"""
UI components module containing all Streamlit rendering logic.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Any
import logging
import os
from utils.document_processor import DocumentProcessor # Assume exists
from services.vector_store import VectorStoreService   # Assume exists
from langchain_core.documents import Document          # Assume exists
from config.settings import AVAILABLE_EMBEDDING_MODELS # Assume exists
from utils.logger_config import get_logger
import concurrent.futures
import functools
import asyncio
import base64
import re  # For regex matching image paths

logger = get_logger(__name__)

def convert_local_images_to_base64(markdown_text: str) -> str:
    """
    Traverse Markdown, replace local image paths with Base64 embeds.
    Assumes Windows-style absolute paths.
    """
    # Internal helper to convert images
    def _replace_image_path(match):
        alt_text = match.group(1)
        image_path = match.group(2)

        # Heuristic: treat drive or leading slash as local path
        if re.match(r'^[A-Za-z]:[\\\/]|^[\\\/]', image_path):
            try:
                # Read image and convert to Base64
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

                # Guess MIME type
                mime_type = "image/jpeg"
                if image_path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif image_path.lower().endswith(".gif"):
                    mime_type = "image/gif"

                # Return Base64 markdown
                return f'![{alt_text}](data:{mime_type};base64,{encoded_string})'
            except FileNotFoundError:
                st.warning(f"Image file not found, cannot embed: {image_path}")
                return match.group(0)
            except Exception as e:
                st.error(f"Error processing image '{image_path}': {e}")
                return match.group(0)
        else:
            # Non-local path, keep as is
            return match.group(0)

    # Regex to find markdown images
    pattern = r'!\[(.*?)\]\((.*?)\)'
    corrected_markdown_text = re.sub(pattern, _replace_image_path, markdown_text)

    return corrected_markdown_text


def run_async_in_thread(coro):
    """
    Run a coroutine in a new thread and wait for it to complete.
    This avoids issues with directly manipulating the loop in an existing event loop (like Streamlit's).
    Returns the coroutine result or raises an exception.
    """
    def _run_in_thread():
        # Create and run a new event loop in the new thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run coroutine in this new loop until complete
            return loop.run_until_complete(coro)
        finally:
            # Cleanup: close the newly created loop
            loop.close()
            asyncio.set_event_loop(None)  # Reset thread's event loop

    # Use ThreadPoolExecutor to execute _run_in_thread function in a new thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit task and wait for result
        future = executor.submit(_run_in_thread)
        # Block waiting for result (this happens in main thread, but won't block Streamlit's event loop too long, as work is in background thread)
        # If there's an exception inside the coroutine, future.result() will re-raise it
        return future.result()


class UIComponents:
    """UI component class, encapsulates all Streamlit UI rendering logic"""

    @staticmethod
    def inject_custom_css():
        """Inject custom CSS"""
        st.markdown("""
            <style>
            /* Global styles */
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            /* Chat container */
            .chat-container {
                height: calc(100vh - 250px); /* Adjust height to fit input box */
                overflow-y: auto;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
                margin-bottom: 20px;
            }
            
            /* Message bubble */
            .message-bubble {
                display: flex;
                align-items: flex-start;
                margin: 15px 0;
                animation: fadeIn 0.3s ease-in;
            }
            
            /* User message */
            .user-message {
                flex-direction: row-reverse;
            }
            
            /* Assistant message */
            .assistant-message {
                flex-direction: row;
            }
            
            /* System message */
            .system-message {
                justify-content: center;
                width: 100%;
            }
            
            /* Avatar */
            .avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                margin: 0 10px;
                flex-shrink: 0;
            }
            
            .user-avatar {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
            }
            
            .assistant-avatar {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: white;
            }
            
            .system-avatar {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
            }
            
            /* Message content */
            .message-content {
                max-width: 70%;
                padding: 15px 20px;
                border-radius: 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                line-height: 1.5;
                position: relative;
                word-wrap: break-word;
            }
            
            .user-message .message-content {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .assistant-message .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-left-radius: 5px;
            }
            
            .system-message .message-content {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
                text-align: center;
                font-size: 14px;
                max-width: 80%;
            }
            
            /* Thinking message */
            .thinking-message {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 20px;
                margin: 15px 0;
                text-align: center;
                animation: pulse 2s infinite;
                max-width: 300px;
                margin: 15px auto;
            }
            
            /* Bottom input */
            .input-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                backdrop-filter: blur(10px);
                border-top: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.1);
                z-index: 1000;
            }
            
            /* Input */
            .stTextInput > div > div > input {
                padding: 15px 20px;
                border-radius: 25px;
                border: 2px solid #e0e0e0;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #4facfe;
                box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
            }
            
            /* Send button */
            .stButton > button {
                height: 50px;
                border-radius: 25px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                font-weight: bold;
                transition: transform 0.2s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            /* Scrollbar */
            .chat-container::-webkit-scrollbar {
                width: 8px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
            }
            
            .chat-container::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.5);
            }
            
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
            }
            
            [data-testid="stSidebar"] .stMarkdown {
                color: white;
            }
            
            /* Expander */
            .streamlit-expanderHeader {
                background: rgba(255, 255, 255, 0.1) !important;
                border-radius: 10px !important;
                margin: 5px 0 !important;
            }
            
            .streamlit-expanderContent {
                background: rgba(255, 255, 255, 0.05) !important;
                border-radius: 10px !important;
                padding: 15px !important;
            }

            /* Empty state */
            .no-messages {
                text-align: center;
                color: #999;
                font-style: italic;
                margin-top: 50px;
            }
                    
            .chat-container:empty {
                 height: auto;
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def scroll_to_bottom():
        """Auto scroll to bottom"""
        st.components.v1.html(
            """
            <script>
                var chatContainer = parent.document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            </script>
            """,
            height=0
        )

    # 1. Render model selectors
    @staticmethod
    def render_model_selection(available_models: List[str], current_model: str, embedding_models: List[str],
                               current_embedding_model: str) -> Tuple[str, str]:
        """
        available_models - model list
        current_model - selected model
        embedding_models - embedding list
        current_embedding_model - selected embedding

        @return (chosen model, chosen embedding)
        """
        st.sidebar.header("⚙️ Settings")

        new_model = st.sidebar.selectbox(
            "Choose model",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="Select the language model"
        )

        new_embedding_model = st.sidebar.selectbox(
            "Embedding model",
            options=embedding_models,
            index=embedding_models.index(current_embedding_model) if current_embedding_model in embedding_models else 0,
            help="Select embedding model for documents"
        )

        return new_model, new_embedding_model

    # 2. Render RAG settings
    @staticmethod
    def render_rag_settings(rag_enabled: bool, similarity_threshold: float, default_threshold: float) -> Tuple[bool, float]:
        """
        rag_enabled - RAG enabled
        similarity_threshold - similarity threshold
        default_threshold - default threshold

        @return (RAG enabled, similarity threshold)
        """
        st.sidebar.subheader("RAG Settings")

        new_rag_enabled = st.sidebar.checkbox(
            "Enable RAG",
            value=rag_enabled,
            help="Use uploaded docs to enhance answers"
        )

        new_similarity_threshold = st.sidebar.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=similarity_threshold,
            step=0.05,
            help="Higher = stricter matching"
        )

        # Reset threshold
        if st.sidebar.button("Reset threshold", use_container_width=True):
            new_similarity_threshold = default_threshold

        return new_rag_enabled, new_similarity_threshold

    # 3. Render chat stats
    @staticmethod
    def render_chat_stats(chat_history):
        """
        chat_history - history manager
        """
        st.sidebar.header("💬 Conversation history")
        stats = chat_history.get_stats()
        st.sidebar.info(f"Total messages: {stats['total_messages']} User messages: {stats['user_messages']}")

        if st.sidebar.button("📥 Export history", use_container_width=True):
            csv = chat_history.export_to_csv()
            if csv:
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Assume export service exists
        try:
            from services.export_file import mdcontent2docx, mdcontent2pdf
            if st.sidebar.button("✅ Export latest answer to PDF", use_container_width=True):
                last_assistant_message = None
                for message in reversed(chat_history.history):
                    if message.get('role') == 'assistant':
                        last_assistant_message = message.get('content', '')
                        break
                if last_assistant_message:
                    mdcontent2pdf(last_assistant_message, './now_content.pdf')
                    st.sidebar.success("PDF exported")

            if st.sidebar.button("🚀 Export latest answer to DOCX", use_container_width=True):
                last_assistant_message = None
                for message in reversed(chat_history.history):
                    if message.get('role') == 'assistant':
                        last_assistant_message = message.get('content', '')
                        break
                if last_assistant_message:
                    mdcontent2docx(last_assistant_message, './now_content.docx')
                    st.sidebar.success("DOCX exported")
        except ImportError:
             st.sidebar.warning("Export service not configured")

        if st.sidebar.button("✨ Clear conversation", use_container_width=True):
            chat_history.clear_history()
            st.rerun()

    # 4. Render document upload
    @staticmethod
    def render_document_upload(
            document_processor: DocumentProcessor,
            vector_store: VectorStoreService,
            processed_documents: List[str]
    ) -> Tuple[List[Document], VectorStoreService]:
        """
        document_processor - processor
        vector_store - vector service
        processed_documents - processed list

        @return (all_docs, vector_store)
        """
        with st.expander("📁 Upload documents to build the knowledge base", expanded=not bool(processed_documents)):
            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, DOCX, MD",
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True
            )

            if not vector_store.vector_store:
                st.warning("⚠️ Configure vector store in the sidebar to enable processing.")

            all_docs = []
            if uploaded_files:
                if st.button("Process Documents"):
                    with st.spinner("Processing documents..."):
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name not in processed_documents:
                                try:
                                    # Handle all file types
                                    print(uploaded_file)
                                    result = document_processor.process_file(uploaded_file)

                                    if isinstance(result, list):
                                        # Document list (PDF)
                                        all_docs.extend(result)
                                    elif isinstance(result, Document):
                                        # Single Document with path metadata
                                        all_docs.append(result)
                                    else:
                                        # Text content without metadata; fall back to filename
                                        doc = Document(
                                            page_content=str(result),
                                            metadata={"source": uploaded_file.name}
                                        )
                                        all_docs.append(doc)

                                    processed_documents.append(uploaded_file.name)
                                    st.success(f"✅ Processed: {uploaded_file.name}")
                                except Exception as e:
                                    st.error(f"❌ Failed: {uploaded_file.name} - {str(e)}")
                            else:
                                st.warning(f"⚠️ Already exists: {uploaded_file.name}")

                    if all_docs:
                        document_paths = []
                        for doc in all_docs:
                            # Extract path from metadata - prefer file_path (full path) over source (may be just filename)
                            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                                # Try file_path first (full path), then fall back to source
                                file_path = doc.metadata.get('file_path') or doc.metadata.get('source')
                                if file_path and isinstance(file_path, str):
                                    # Ensure it's a valid path
                                    if os.path.exists(file_path):
                                        document_paths.append(file_path)
                                        logger.debug(f"Added document path: {file_path}")
                                    else:
                                        # Try relative path from current directory
                                        if not os.path.isabs(file_path):
                                            # Check if it's in static/uploads
                                            possible_paths = [
                                                os.path.join("static", "uploads", file_path),
                                                file_path
                                            ]
                                            found = False
                                            for possible_path in possible_paths:
                                                if os.path.exists(possible_path):
                                                    document_paths.append(possible_path)
                                                    logger.debug(f"Found document at: {possible_path}")
                                                    found = True
                                                    break
                                            if not found:
                                                st.warning(f"⚠️ File not found: {file_path}")
                                                logger.warning(f"Cannot find file: {file_path}")
                                        else:
                                            st.warning(f"⚠️ File not found: {file_path}")
                                            logger.warning(f"Cannot find file: {file_path}")
                                else:
                                    st.warning(f"⚠️ Cannot get file path from document: {doc}")
                                    logger.warning(f"Cannot get file path from document: {doc}")
                            else:
                                st.warning(f"⚠️ Document missing valid metadata: {doc}")
                                logger.warning(f"Document missing valid metadata: {doc}")

                        if not document_paths:
                            st.error("❌ No valid document paths to process.")
                            logger.error("No valid document paths to process.")
                            return all_docs, vector_store
                        with st.spinner("Building vector index..."):
                            success = run_async_in_thread(vector_store.create_vector_store(document_paths))

            # Display processed documents list
            if processed_documents:
                st.subheader("Processed documents")
                for doc in processed_documents:
                    st.markdown(f"- {doc}")

                if st.button("Clear all documents"):
                    with st.spinner("Clearing vector index..."):
                        vector_store.clear_index()
                        processed_documents.clear()
                    st.success("✅ All documents cleared")
                    st.rerun()

            return all_docs, vector_store

    # 5. Render chat history
    @staticmethod
    def render_chat_history(chat_history):
        """
        chat_history - history manager
        """
        # Chat container
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            # Render messages
            for message in chat_history.history:
                role = message.get('role', '')
                content = message.get('content', '')

                if role == "assistant_think":
                    with st.expander("💡 View reasoning <think> ... </think>"):
                        st.markdown(content)
                elif role == "retrieved_doc":
                    with st.expander(f"📖 Retrieved document chunks", expanded=False):
                        if isinstance(content, list):
                            for idx, doc in enumerate(content, 1):
                                st.markdown(f"**Chunk {idx}:**\n{doc}")
                        else:
                            st.markdown(content)
                else:
                    corrected_markdown_content = convert_local_images_to_base64(content)

                    # Render role-specific bubbles
                    if role == "user":
                        st.markdown(f'''
                            <div class="message-bubble user-message">
                                <div class="avatar user-avatar">👤</div>
                                <div class="message-content">{corrected_markdown_content}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    elif role == "assistant":
                        st.markdown(f'''
                            <div class="message-bubble assistant-message">
                                <div class="avatar assistant-avatar">🤖</div>
                                <div class="message-content">{corrected_markdown_content}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    elif role == "system":
                        st.markdown(f'''
                            <div class="message-bubble system-message">
                                <div class="avatar system-avatar">ℹ️</div>
                                <div class="message-content">{corrected_markdown_content}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    # Note: role == "thinking" is handled in app.py

            # Empty state
            if not chat_history.history:
                st.markdown('<div class="no-messages">No messages yet</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # 自动滚动到底部
        UIComponents.scroll_to_bottom()

    # 6. Render input area (callback clears input)
    @staticmethod
    def render_input_area() -> None:
        """
        Render bottom input box.
        Use callback to avoid session_state conflicts.
        """
        # --- Callback ---
        def on_send_click():
            """
            Called when send is clicked.
            Stores current input to a temp session_state field, then clears.
            """
            # 1. Get current input
            current_input = st.session_state.get("user_input", "").strip()

            # 2. Save pending input
            if current_input:
                st.session_state._input_to_process = current_input

            # 3. Clear field
            st.session_state.user_input = ""

        # --- UI ---
        # Bottom-fixed input area
        input_container = st.container()
        with input_container:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)

            # Layout
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_input(
                    "User input",
                    placeholder="Enter your question...",
                    label_visibility="collapsed",
                    key="user_input"
                )
            with col2:
                st.button(
                    "Send",
                    use_container_width=True,
                    key="send_button",
                    on_click=on_send_click
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # Input handling is done via callbacks; app.run consumes the temp value

# 确保 __name__ == "__main__" 不会执行任何代码，因为这是被导入的模块
if __name__ == "__main__":
    pass  # UIComponents 通常不直接运行
