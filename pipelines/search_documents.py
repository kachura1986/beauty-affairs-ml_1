import asyncio

from dependencies import PipelineDependencies
from services.search_index.products_search_index import ProductsSearchIndex
from services.search_index.services_search_index import ServicesSearchIndex


async def main() -> None:
    """
    Creates product and service indexes.
    """
    dependencies = await PipelineDependencies.create()
    azure_search_manager = dependencies.azure_search_index_manager

    products_index_instance = ProductsSearchIndex(
        index_name='products_test',
        azure_search_index_manager=azure_search_manager,
        embedder=dependencies.azure_embedder
    )

    services_index_instance = ServicesSearchIndex(
        index_name='services_test',
        azure_search_index_manager=azure_search_manager,
        embedder=dependencies.azure_embedder
    )

    try:
        # Searching products
        print("Searching products")
        products = await products_index_instance.search_documents(
            query_text="Face cream for wrinkles, anti-aging skincare, hydrating moisturizer, collagen serum, and night cream for improving skin firmness and reducing the appearance of fine lines",
            top_k=5,
            vector_candidates=7,
            filters=None
        )

        for product in products:
            print(product)

        # Searching services
        print("Searching services")
        services = await services_index_instance.search_documents(
            query_text="I want to tighten sagging skin on my neck and chin and improve my jawline without surgery or downtime.",
            top_k=5,
            vector_candidates=7,
            filters=None
        )

        for service in services:
            print(service)

    finally:
        await dependencies.close()


if __name__ == '__main__':
    asyncio.run(main())
