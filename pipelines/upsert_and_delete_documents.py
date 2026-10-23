import asyncio

from dependencies import PipelineDependencies
from services.search_index.products_search_index import ProductsSearchIndex
from services.search_index.services_search_index import ServicesSearchIndex

from utilities.file_io import FileIO


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

    products_path = 'data/products_services/products_test.json'
    products = list(FileIO.load_json(products_path)[0]['data'])

    services_path = 'data/products_services/services_test.json'
    services = list(FileIO.load_json(services_path)[0]['data'])

    try:
        # Upsert products and services
        print("Upserting products and services")
        await products_index_instance.upsert_documents(documents=products)
        await services_index_instance.upsert_documents(documents=services)

        await asyncio.sleep(3)

        products_document_count = await products_index_instance.get_document_count()
        services_document_count = await services_index_instance.get_document_count()

        print(f"Количество документов в продуктах: {products_document_count}")
        print(f"Количество документов в услугах: {services_document_count}")

        # Delete products and services
        print("Deleting products and services")
        await products_index_instance.delete_documents(documents=products)
        await services_index_instance.delete_documents(documents=services)

        products_document_count = await products_index_instance.get_document_count()
        services_document_count = await services_index_instance.get_document_count()

        print(f"Количество документов в продуктах: {products_document_count}")
        print(f"Количество документов в услугах: {services_document_count}")

    finally:
        await dependencies.close()


if __name__ == '__main__':
    asyncio.run(main())
