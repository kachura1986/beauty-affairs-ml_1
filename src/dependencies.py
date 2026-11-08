import asyncio
from dataclasses import dataclass

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from fastapi import Request
from openai import AsyncOpenAI

from core.config import settings
from core.logger import get_logger
from services.agent.config import DEFAULT_AGENT_KEY, load_agent_configs, require_default_agent
from services.agent.context import SYSTEM_PROMPT
from services.agent.tools import create_product_tools
from services.embedding import AzureOpenAIEmbedder
from services.llm import AzureOpenAIModel
from services.search_index import AzureSearchIndexManager, ProductsSearchIndex
from services.text_normalization import TextNormalizer
from use_cases import AgentService, IndexService


# DI container
@dataclass
class PipelineDependencies:
    agent_services: dict[str, AgentService]
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
                api_key=settings.AZURE_OPENAI_API_KEY.get_secret_value(),
                # The only time limit is asyncio.wait_for(remaining) in AzureOpenAIModel.chat,
                # driven by the agent Budget. SDK defaults here (600s timeout, 2 retries)
                # would race with that budget and raise a different exception type.
                timeout=None,
                max_retries=0,
            )

            azure_embedder = AzureOpenAIEmbedder(
                client=openai_client,
                model_name=settings.AZURE_OPENAI_EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSIONS,
                normalizer=TextNormalizer(),
            )

            search_credential = AzureKeyCredential(settings.SEARCH_API_KEY.get_secret_value())

            azure_search_index_client = SearchIndexClient(endpoint=settings.SEARCH_ENDPOINT,
                                                          credential=search_credential)

            azure_search_index_manager = AzureSearchIndexManager(search_index_client=azure_search_index_client)

            products_index = ProductsSearchIndex(
                index_name='products_test',
                azure_search_index_manager=azure_search_index_manager,
                embedder=azure_embedder
            )

            agent_configs = load_agent_configs(settings.AGENTS_CONFIG)
            # Fail-fast validation AND explicit capture: get_agent_service() relies on
            # DEFAULT_AGENT_KEY being present in agent_services.
            default_agent = require_default_agent(agent_configs, settings.AGENTS_CONFIG)  # noqa: F841

            shared_tools = create_product_tools(
                products_index,
                default_timeout_seconds=settings.TOOL_DEFAULT_TIMEOUT_SECONDS,
                max_output_chars=settings.TOOL_DEFAULT_MAX_OUTPUT_CHARS,
            )
            agent_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_AGENT_RUNS)

            agent_services: dict[str, AgentService] = {}

            for name, cfg in agent_configs.items():
                model = AzureOpenAIModel(
                    model_name=cfg.model.model_name,
                    azure_openai_client=openai_client,
                    temperature=cfg.model.temperature,
                    max_tokens=cfg.model.max_tokens,
                    input_cost_per_million=cfg.model.input_cost_per_million,
                    output_cost_per_million=cfg.model.output_cost_per_million,
                )

                agent_services[name] = AgentService(
                    llm=model,
                    tools=shared_tools,
                    system_prompt=SYSTEM_PROMPT,
                    runs_dir=settings.RUNS_DIR,
                    policy=cfg.policy,
                    budget_limits=cfg.budget,
                    semaphore=agent_semaphore,
                    agent_name=name,
                )

            logger.info(
                "Created %d agent service(s): %s",
                len(agent_services),
                sorted(agent_services),
            )

            index_service = IndexService(search_index_manager=azure_search_index_manager, embedder=azure_embedder)

            return cls(
                agent_services=agent_services,
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
    # Invariant: PipelineDependencies.create() calls require_default_agent(),
    # so DEFAULT_AGENT_KEY is always present here. Validated at startup, not per-request.
    return get_pipeline_dependencies(request).agent_services[DEFAULT_AGENT_KEY]


# Index
def get_index_service(request: Request) -> IndexService:
    return get_pipeline_dependencies(request).index_service
