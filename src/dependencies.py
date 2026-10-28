import logging
from dataclasses import dataclass
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from fastapi import Request
from openai import AsyncOpenAI

from services.agent.context import SYSTEM_PROMPT
from core.config import settings
from core.logger import setup_logging
from services.agent.tools import create_product_tools
from services.embedding import AzureOpenAIEmbedder
from services.llm import AzureOpenAIModel
from services.search_index import AzureSearchIndexManager, ProductsSearchIndex
from services.text_normalization.text_normalizer import TextNormalizer
from use_cases import AgentService, IndexService


# Logger
@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    return setup_logging(settings.LOGS_DIR, 'app.log')


# DI container
@dataclass
class PipelineDependencies:
    agent_service: AgentService
    azure_embedder: AzureOpenAIEmbedder
    azure_search_index_client: SearchIndexClient
    azure_search_index_manager: AzureSearchIndexManager
    index_service: IndexService
    openai_client: AsyncOpenAI

    @classmethod
    async def create(cls) -> "PipelineDependencies":
        logger = get_logger()
        openai_client = None
        azure_search_index_client = None

        try:
            openai_client = AsyncOpenAI(
                base_url=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                timeout=20.0,
                max_retries=2,
            )

            azure_model = AzureOpenAIModel(
                model_name=settings.AZURE_OPENAI_CHAT_MODEL,
                azure_openai_client=openai_client,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                input_cost_per_million=settings.INPUT_COST_PER_MILLION,
                output_cost_per_million=settings.OUTPUT_COST_PER_MILLION,
            )

            azure_embedder = AzureOpenAIEmbedder(
                client=openai_client,
                model_name=settings.AZURE_OPENAI_EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSIONS,
                normalizer=TextNormalizer(),
            )

            search_credential = AzureKeyCredential(settings.SEARCH_API_KEY)

            azure_search_index_client = SearchIndexClient(endpoint=settings.SEARCH_ENDPOINT,
                                                          credential=search_credential)

            azure_search_index_manager = AzureSearchIndexManager(search_index_client=azure_search_index_client)

            # TODO: Create products and services search index for both centers
            products_index = ProductsSearchIndex(
                index_name='products_test',
                azure_search_index_manager=azure_search_index_manager,
                embedder=azure_embedder
            )

            # Use cases
            agent_service = AgentService(
                llm=azure_model,
                tools=create_product_tools(products_index),
                system_prompt=SYSTEM_PROMPT,
                runs_dir=settings.RUNS_DIR,
            )
            index_service = IndexService(search_index_manager=azure_search_index_manager, embedder=azure_embedder)

            return cls(
                agent_service=agent_service,
                azure_embedder=azure_embedder,
                azure_search_index_client=azure_search_index_client,
                azure_search_index_manager=azure_search_index_manager,
                index_service=index_service,
                openai_client=openai_client,
            )

        except BaseException:
            logger.exception("Failed to initialize pipeline dependencies")

            if azure_search_index_client is not None:
                try:
                    await azure_search_index_client.close()
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
                ("Azure Search client", self.azure_search_index_manager),
                ("Azure OpenAI client", self.openai_client),
        ):
            try:
                await resource.close()
            except Exception:
                logger.exception("Failed to close %s", name)


# DI container
def get_pipeline_dependencies(request: Request) -> PipelineDependencies:
    return request.app.state.dependencies


# Dependency providers
# Agent
def get_agent_service(request: Request) -> AgentService:
    return get_pipeline_dependencies(request).agent_service


# Index
def get_index_service(request: Request) -> IndexService:
    return get_pipeline_dependencies(request).index_service
