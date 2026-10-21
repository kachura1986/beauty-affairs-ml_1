from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncAzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexClient

from application import AgentService
from core.config import settings
from dependencies import get_logger
from services.agent.tools import DEMO_TOOLS
from services.embedding import AzureOpenAIEmbedder
from services.llm import AzureOpenAIModel
from services.text_normalization.text_normalizer import TextNormalizer
from services.search.azure_search_manager import AzureSearchManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the application lifecycle.
    Ensures that setup and cleanup occur only within the worker process.
    """
    logger = get_logger()
    logger.info("Starting %s server...", settings.APP_TITLE)

    openai_client = None
    search_client = None
    azure_search_manager = None

    try:
        # Azure OpenAI client
        openai_client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT.rstrip("/"),
            api_version=settings.OPENAI_API_VERSION,
            timeout=20.0,
            max_retries=0,
        )

        app.state.azure_openai_client = openai_client

        # LLM
        app.state.azure_model = AzureOpenAIModel(
            model_name=settings.AZURE_OPENAI_CHAT_MODEL,
            azure_openai_client=openai_client,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

        # Embedding
        app.state.azure_embedder = AzureOpenAIEmbedder(
            client=openai_client,
            model_name=settings.AZURE_OPENAI_EMBEDDING_MODEL,
            dimensions=1024,
            normalizer=TextNormalizer(),
        )

        # Agent
        app.state.agent_service = AgentService(llm=app.state.azure_model, tools=DEMO_TOOLS)

        # Azure AI Search
        search_credential = AzureKeyCredential(settings.SEARCH_API_KEY)
        search_client = SearchIndexClient(endpoint=settings.SEARCH_ENDPOINT, credential=search_credential)
        azure_search_manager = AzureSearchManager(
            search_index_client=search_client,
            embeddings_client=app.state.azure_embedder,
        )

        app.state.search_credential = search_credential
        app.state.search_index_client = search_client
        app.state.azure_search_manager = azure_search_manager

        logger.info("Application startup completed.")

        yield

    finally:
        logger.info("Shutting down FastAPI server...")

        if azure_search_manager is not None:
            try:
                await azure_search_manager.close()
            except Exception:
                logger.exception("Failed to close Azure Search Manager")

        if search_client is not None:
            try:
                await search_client.close()
            except Exception:
                logger.exception("Failed to close Azure Search client")

        if openai_client is not None:
            try:
                await openai_client.close()
            except Exception:
                logger.exception("Failed to close Azure OpenAI client")

        logger.info("Application shutdown completed.")
