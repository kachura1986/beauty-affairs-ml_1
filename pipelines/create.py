import asyncio
import json
import logging
from pathlib import Path

from dependencies import PipelineDependencies

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

    dependencies = await PipelineDependencies.create()

    try:
        logger.info("Creating test product and service indexes")

        await dependencies.azure_search_manager.create_indexes(
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
        await dependencies.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
