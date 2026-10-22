import logging
from dataclasses import dataclass
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from fastapi import Request
from openai import AsyncAzureOpenAI

from application import AgentService
from core.config import settings
from core.logger import setup_logging
from services.agent.tools import DEMO_TOOLS
from services.embedding import AzureOpenAIEmbedder
from services.llm import AzureOpenAIModel, LLMModel
from services.search_index.azure_index_manager import AzureSearchManager
from services.text_normalization.text_normalizer import TextNormalizer


# Logger
@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    return setup_logging('app.log')


# DI container
@dataclass
class PipelineDependencies:
    openai_client: AsyncAzureOpenAI
    azure_model: LLMModel
    azure_embedder: AzureOpenAIEmbedder
    agent_service: AgentService
    search_credential: AzureKeyCredential
    search_index_client: SearchIndexClient
    azure_index_manager: AzureSearchManager

    @classmethod
    async def create(cls) -> "PipelineDependencies":
        logger = get_logger()
        openai_client = None
        search_index_client = None
        index_manager = None

        try:
            openai_client = AsyncAzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT.rstrip("/"),
                api_version=settings.OPENAI_API_VERSION,
                timeout=20.0,
                max_retries=0,
            )

            azure_model = AzureOpenAIModel(
                model_name=settings.AZURE_OPENAI_CHAT_MODEL,
                azure_openai_client=openai_client,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )

            azure_embedder = AzureOpenAIEmbedder(
                client=openai_client,
                model_name=settings.AZURE_OPENAI_EMBEDDING_MODEL,
                dimensions=1024,
                normalizer=TextNormalizer(),
            )

            agent_service = AgentService(llm=azure_model, tools=DEMO_TOOLS)

            search_credential = AzureKeyCredential(settings.SEARCH_API_KEY)

            search_index_client = SearchIndexClient(endpoint=settings.SEARCH_ENDPOINT, credential=search_credential)

            index_manager = AzureSearchManager(search_index_client=search_index_client)

            return cls(
                openai_client=openai_client,
                azure_model=azure_model,
                azure_embedder=azure_embedder,
                agent_service=agent_service,
                search_credential=search_credential,
                search_index_client=search_index_client,
                azure_index_manager=index_manager,
            )

        except BaseException:
            logger.exception("Failed to initialize pipeline dependencies")

            if search_index_client is not None:
                try:
                    await search_index_client.close()
                except Exception:
                    logger.exception("Failed to close search_index client")

            if openai_client is not None:
                try:
                    await openai_client.close()
                except Exception:
                    logger.exception("Failed to close OpenAI client")

            raise

    async def close(self) -> None:
        logger = get_logger()

        for name, resource in (
                ("Azure Search client", self.search_index_client),
                ("Azure OpenAI client", self.openai_client),
        ):
            try:
                await resource.close()
            except Exception:
                logger.exception("Failed to close %s", name)


# Dependency providers
def get_pipeline_dependencies(request: Request) -> PipelineDependencies:
    return request.app.state.dependencies


# Agent
def get_agent_service(request: Request) -> AgentService:
    return get_pipeline_dependencies(request).agent_service
