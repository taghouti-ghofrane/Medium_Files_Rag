# -*- coding: utf-8 -*-
# @Time    : 2025/08/06 10:00:00
# @Author  : nyzhhd
# @File    : app.py
# @Description: Main application file

from datetime import datetime
import logging
import re
from config.settings import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    AVAILABLE_EMBEDDING_MODELS,
    MAX_HISTORY_TURNS
)
# RAGAgent: handles user input and produces responses, wrapping model logic.
from models.agent import RAGAgent
# ChatHistoryManager: manages conversation history
from utils.chat_history import ChatHistoryManager
# DocumentProcessor: processes uploaded documents
from utils.document_processor import DocumentProcessor
# VectorStoreService: vector DB service for indexing and search
from services.vector_store import VectorStoreService

from utils.decorators import error_handler, log_execution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class agent_response:
    """
    RAG application main class
    """
    def __init__(self):
        """
        @description Initialize application
        """
        self.chat_history = ChatHistoryManager()  # chat history manager
        self.document_processor = DocumentProcessor()  # document processor
        self.vector_store = VectorStoreService()  # vector store service
        self.rag_enabled = True
        self.model = DEFAULT_MODEL
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        self.max_turns = MAX_HISTORY_TURNS
        self.human_content = None
        self.agent = RAGAgent(self.model) 
        logger.info("Application initialized successfully")
    
    def update(self, rag_enabled , model, similarity_threshold, max_turns = 5):
        self.rag_enabled = rag_enabled
        
        if(self.model != model):
            self.model = model
            self.agent = RAGAgent(self.model)
            logger.info("Model updated successfully")
        self.similarity_threshold = similarity_threshold
        self.max_turns = max_turns
        logger.info("Application parameters updated successfully")
    
    async def process_user_input(self, prompt: str):
        """
        prompt - user prompt text

        1️⃣ RAG mode: retrieve docs → build context → call model
        2️⃣ Plain mode: call model directly
        """
        
        self.chat_history.add_message("user", prompt)
        if self.rag_enabled:
            await self._process_rag_query(prompt)  # RAG path
        else:
            await self._process_simple_query(prompt)  # plain path
    
    
    # 5. Process RAG query
    async def _process_rag_query(self, prompt: str):
        """
        prompt - user prompt text
        """
        top_k = 3
        docs = await self.vector_store.search_documents(  
            prompt,
            top_k,
            self.similarity_threshold
        )
        logger.info(f"Documents retrieved: {len(docs)}")  
        # Build document context
        context = self.vector_store.get_context(docs)  
        # Create agent and run
        response = self.agent.run(  
            prompt, 
            context=context,
            history = self.chat_history.get_formatted_history(2),
            isapp = True
        )
        # Handle response
        await self._process_response(response, docs)  
    

    # 6. Process simple query
    async def _process_simple_query(self, prompt: str):
        """
        prompt - user prompt text
        """
         
        # Run agent to get response
        history = self.chat_history.get_formatted_history(2)
        response = self.agent.run(prompt, context = None, history = history,  isapp = True)  
        # Handle response
        await self._process_response(response)  
    
    
    # 7. Handle agent response
    async def _process_response(self, response: str, docs=None):
        """
        response - raw model response
        docs - retrieved documents (optional)
        """
        # 7.1 Parse think block
        think_pattern = r'<think>([\s\S]*?)</think>'
        think_match = re.search(think_pattern, response)
        if think_match:
            think_content = think_match.group(1).strip()
            response_wo_think = re.sub(think_pattern, '', response).strip()
        else:
            think_content = None
            response_wo_think = response
        human_pattern = r'<human>([\s\S]*?)</human>'  # pattern for human handoff
        human_match = re.search(human_pattern, response_wo_think)
        if human_match:
            human_content = human_match.group(1).strip()
            response_wo_think = re.sub(human_pattern, '', response_wo_think).strip()
        else:
            human_content = None
            response_wo_think = response_wo_think
        
        if human_content:
            self.human_content = human_content
        else:
            self.human_content = None
        
        # 7.2 Save response to history
        self.chat_history.add_message("assistant", response_wo_think)
        if think_content:
            self.chat_history.add_message("assistant_think", think_content)
        if docs:
            doc_contents = [doc.get('content', '') for doc in docs]  # avoid KeyError
            self.chat_history.add_message("retrieved_doc", doc_contents)
        
        # Save last response
        self.last_response = response_wo_think


    # 获取最后的响应
    def get_last_response(self):
        """Get last AI response"""
        return getattr(self, 'last_response', "Sorry, I didn't understand your question.")
    
    # 清除响应历史
    def clear_response(self):
        """Clear response history"""
        if hasattr(self, 'last_response'):
            delattr(self, 'last_response')
    
    # 获取历史记录
    def get_history(self):
        """Get conversation history"""
        return self.chat_history.get_formatted_history(self.max_turns)
    
    def get_history_summary(self):
        """Get conversation summary"""
        response = self.agent.summary_chat_histroy(self.get_history())
        # print("history: ",response)
        think_pattern = r'<think>([\s\S]*?)</think>'
        think_match = re.search(think_pattern, response)
        if think_match:
            think_content = think_match.group(1).strip()
            summary_histroy = re.sub(think_pattern, '', response).strip()
        else:
            think_content = None
            summary_histroy = response
        return summary_histroy


    # Entry: run application
    async def run(self, prompt): 
        # print("history: ", self.get_history_summary())  
        if prompt:
            await self.process_user_input(prompt)
            return self.get_last_response()
        return "Please enter a valid question."
    
    # Determine if we should hand off to a human; return summary if so
    def is_to_human(self):
        if not self.human_content:
            return False, None
        else:
            return True, self.get_history_summary()
                
        
import asyncio
if __name__ == "__main__":
    app = agent_response()
    response = asyncio.run(app.run('request human'))
    print(f"Response: {response}")