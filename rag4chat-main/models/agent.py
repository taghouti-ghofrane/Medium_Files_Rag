# Full updated RAGAgent implementation with .env support
"""
Agent model - OpenAI only (updated for .env loading)
"""
from typing import Optional
import logging
import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.tools.reasoning import ReasoningTools
from agno.tools.function import Function

from config.settings import (
    DEFAULT_MODEL,
    AMAP_API_KEY,
    DB_PATH,
)
from services.weather_tools import WeatherTools

# Load .env variables
load_dotenv()

# Read OpenAI variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

from utils.logger_config import get_logger
logger = get_logger(__name__)

# Try importing OpenAI support from Agno
OPENAI_AVAILABLE = False
OpenAI = None

try:
    from agno.models.openai import OpenAIChat as OpenAI
    OPENAI_AVAILABLE = True
    logger.info("Using OpenAIChat from agno.models.openai")
except ImportError:
    try:
        from agno.models import OpenAI
        OPENAI_AVAILABLE = True
        logger.info("Using OpenAI from agno.models")
    except ImportError:
        try:
            from openai import OpenAI as OpenAIClient
            from agno.models.base import Model

            class OpenAIWrapper(Model):
                """Fallback wrapper when Agno does not bundle OpenAI"""
                def __init__(self, id: str, api_key: str, base_url: str):
                    super().__init__()
                    self.id = id
                    self.client = OpenAIClient(api_key=api_key, base_url=base_url)

                def complete(self, messages, **kwargs):
                    response = self.client.chat.completions.create(
                        model=self.id,
                        messages=messages,
                        **kwargs
                    )
                    return response.choices[0].message.content

            OpenAI = OpenAIWrapper
            OPENAI_AVAILABLE = True
            logger.info("Using OpenAI wrapper (fallback)")

        except ImportError:
            OPENAI_AVAILABLE = False
            logger.error("OpenAI not available. Install with: pip install openai")


class RAGAgent:
    """RAG agent class wrapping model interactions"""

    def __init__(self, model_version: str = DEFAULT_MODEL):
        self.model_version = model_version
        self.agent = self._create_agent()

    def _has_valid_openai_key(self) -> bool:
        """Validate OpenAI key from .env"""
        return bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))

    def _create_model(self):
        """Instantiate the OpenAI model"""
        logger.info(f"[Agent] Creating OpenAI model: {self.model_version}")
        
        if not OPENAI_AVAILABLE:
            logger.error("[Agent] ❌ OpenAI support is not available")
            raise ValueError(
                "OpenAI support is missing. Try: pip install --upgrade agno openai"
            )

        if not self._has_valid_openai_key():
            logger.error("[Agent] ❌ Invalid or missing OpenAI API key")
            raise ValueError(
                "Missing or invalid OPENAI_API_KEY in .env (must start with sk-)."
            )

        logger.info(f"[Agent] ✅ Using OpenAI model: {self.model_version}")
        logger.debug(f"[Agent] Base URL: {OPENAI_BASE_URL}")

        return OpenAI(
            id=self.model_version,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    def _create_agent(self) -> Agent:
        """Build the Agno Agent including tools"""

        weather_tools = WeatherTools(AMAP_API_KEY)

        query_weather_function = Function(
            name="query_weather",
            description="Query weather for the specified city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
            },
            entrypoint=weather_tools.query_weather,
        )

        model = self._create_model()

        return Agent(
            name="RAG Agent",
            model=model,
            instructions="""
You are a helpful assistant.
- In RAG mode, answer strictly from the provided documents.
- Otherwise, rely on general reasoning.
- For weather queries: always call query_weather.
            """,
            tools=[
                ReasoningTools(add_instructions=True),
                query_weather_function,
            ],
            markdown=True,
        )

    def run(self, prompt: str, context: Optional[str] = None, history: str = None, isapp: bool = False) -> str:
        """Run agent inference"""
        logger.info(f"[Agent Run] Starting agent inference - Prompt length: {len(prompt)} chars")
        if context:
            logger.debug(f"[Agent Run] Context provided: {len(context)} characters")
        if history:
            logger.debug(f"[Agent Run] History provided: {len(history)} characters")
        logger.debug(f"[Agent Run] isapp mode: {isapp}")

        if isapp:
            if history:
                if context:
                    full_prompt = f"""
[Retrieved Content]
{context}

[User Question]
{prompt}

[Conversation History]
{history}

You are an intelligent customer service assistant. Please remain polite, friendly, and patient. And please answer strictly according to the retrieved content.
                    """
                else:
                    full_prompt = f"""
[User Question]
{prompt}

[Conversation History]
{history}

You are an intelligent customer service assistant. Please remain polite, friendly, and patient.
                    """
            else:
                if context:
                    full_prompt = f"""
[Retrieved Content]
{context}

[User Question]
{prompt}

You are an intelligent customer service assistant. Please remain polite, friendly, and patient.
                    """
                else:
                    full_prompt = f"""
[User Question]
{prompt}

You are an intelligent customer service assistant. Please remain polite, friendly, and patient.
                    """
        else:
            if context:
                full_prompt = f"""
[Retrieved Content]
{context}

[User Question]
{prompt}

Please answer strictly according to the retrieved content.
                """
            else:
                full_prompt = f"""
[User Question]
{prompt}

Please provide accurate and helpful answers.
                """

        logger.info(f"[Agent Run] Executing agent with full prompt ({len(full_prompt)} characters)...")
        response = self.agent.run(full_prompt)
        logger.info(f"[Agent Run] ✅ Response received ({len(response.content)} characters)")
        return response.content

    def summary_chat_histroy(self, context: str) -> str:
        """Summarize chat history"""
        logger.info(f"[Agent Summary] Starting chat history summarization ({len(context)} characters)")
        full_prompt = f"Summarize the following chat history in no more than 300 words:\n[Content]\n{context}\n"
        response = self.agent.run(full_prompt)
        logger.info(f"[Agent Summary] ✅ Summary generated ({len(response.content)} characters)")
        return response.content
