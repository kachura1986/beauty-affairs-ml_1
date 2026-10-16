import logging
from functools import lru_cache

from openai import AsyncAzureOpenAI

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
def get_llm_clients() -> tuple[AsyncAzureOpenAI | None]:
    azure_openai_api_key = settings.AZURE_OPENAI_API_KEY
    azure_openai_endpoint = settings.AZURE_OPENAI_ENDPOINT
    azure_openai_api_version = settings.OPENAI_API_VERSION
    azure_openai_chat_model = settings.AZURE_OPENAI_CHAT_MODEL

    logger = get_logger()


    if not all((
            azure_openai_api_key,
            azure_openai_endpoint,
            azure_openai_api_version,
            azure_openai_chat_model,
    )):
        logger.warning("Azure OpenAI provider is not fully configured; Azure OpenAI provider will be unavailable.")

    azure_openai_client = AsyncAzureOpenAI(
        api_key=azure_openai_api_key,
        azure_endpoint=azure_openai_endpoint.rstrip("/"),
        api_version=azure_openai_api_version,
        timeout=20.0,
        max_retries=0,
    )

    return azure_openai_client,


@lru_cache(maxsize=1)
def get_model_resolver() -> LLMModelResolver:
    azure_openai_client, = get_llm_clients()

    return LLMModelResolver(
        default_temperature=settings.TEMPERATURE,
        default_max_tokens=settings.MAX_TOKENS,
        azure_openai_client=azure_openai_client,
        azure_openai_chat_model=settings.AZURE_OPENAI_CHAT_MODEL,
    )


# Agent
@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService(model_resolver=get_model_resolver(), tools=DEMO_TOOLS)
