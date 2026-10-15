import logging
from functools import lru_cache

from google import genai
from ollama import AsyncClient

from application import AgentService, LLMModelResolver
from core.config import settings
from core.logger import setup_logging
from services.agent.tools import DEMO_TOOLS


# Logger
@lru_cache()
def get_logger() -> logging.Logger:
    return setup_logging('app.log')


# LLM
@lru_cache(maxsize=1)
def get_llm_clients() -> tuple[AsyncClient, genai.Client | None]:
    ollama_host = settings.OLLAMA_HOST
    gemini_api_key = settings.GEMINI_API_KEY

    logger = get_logger()

    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set; Gemini provider will be unavailable.")

    ollama_client = AsyncClient(host=ollama_host)
    genai_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

    return ollama_client, genai_client


@lru_cache(maxsize=1)
def get_model_resolver() -> LLMModelResolver:
    ollama_client, genai_client = get_llm_clients()

    return LLMModelResolver(
        ollama_client=ollama_client,
        genai_client=genai_client,
        ollama_host=settings.OLLAMA_HOST,
        default_temperature=settings.TEMPERATURE,
        default_max_tokens=settings.MAX_TOKENS,
    )


# Agent
@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService(model_resolver=get_model_resolver(), tools=DEMO_TOOLS)
