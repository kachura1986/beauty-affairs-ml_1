import logging
from functools import lru_cache

from openai import AsyncAzureOpenAI

from application import AgentService
from core.config import settings
from core.logger import setup_logging
from services.agent.tools import DEMO_TOOLS
from services.embedding import AzureOpenAIEmbedder
from services.llm import AzureOpenAIModel, LLMModel
from services.text_normalization.text_normalizer import TextNormalizer


# Logger
@lru_cache()
def get_logger() -> logging.Logger:
    return setup_logging('app.log')


# LLM
@lru_cache(maxsize=1)
def azure_openai_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT.rstrip("/"),
        api_version=settings.OPENAI_API_VERSION,
        timeout=20.0,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_azure_model() -> LLMModel:
    return AzureOpenAIModel(
        model_name=settings.AZURE_OPENAI_CHAT_MODEL,
        azure_openai_client=azure_openai_client(),
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )


# Embedding
def get_azure_embedder() -> AzureOpenAIEmbedder:
    return AzureOpenAIEmbedder(
        client=azure_openai_client(),
        model_name=settings.AZURE_OPENAI_EMBEDDING_MODEL,
        dimensions=1024,
        normalizer=TextNormalizer(),
    )


# Agent
@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService(llm=get_azure_model(), tools=DEMO_TOOLS)
