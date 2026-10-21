import asyncio
import json
import logging
import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexClient

from services.embedding import AzureOpenAIEmbedder
from services.search.azure_search_manager import AzureSearchManager

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/products_services")


def load_documents(file_path: Path, prefix: str) -> dict[str, list[dict]]:
    """Load documents from JSON and prepare index names."""
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    documents = data.get("data", {})

    return {
        f"{prefix}_{key}": value
        for key, value in documents.items()
    }


async def main() -> None:
    products = load_documents(DATA_DIR / "products.json", "products")
    services = load_documents(DATA_DIR / "services.json", "services")

    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_api_key = os.environ["AZURE_SEARCH_API_KEY"]
    embedding_model = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]

    openai_client = AsyncAzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT.rstrip("/"),
        api_version=settings.OPENAI_API_VERSION,
        timeout=20.0,
        max_retries=0,
    )

    search_index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_api_key),
    )

    # Используй ту же конфигурацию embedder, что и в приложении.
    embeddings_client = AzureOpenAIEmbedder(
            client=openai_client,
            model_name=embedding_model,
            dimensions=1024,
            normalizer=TextNormalizer(),
        )

    search_manager = AzureSearchManager(
        search_index_client=search_index_client,
        embeddings_client=embeddings_client,
    )

    try:
        logger.info("Creating test product and service indexes")

        await search_manager.create_indexes(
            products=products,
            services=services,
        )

        logger.info(
            "Indexes created and documents uploaded: "
            "%d product indexes, %d service indexes",
            len(products),
            len(services),
        )
    finally:
        await search_manager.close()
        await search_index_client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
