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

    indexes = [ProductsSearchIndex.build_schema('products_test'), ServicesSearchIndex.build_schema('services_test')]

    try:
        print("Creating indexes")
        await azure_search_manager.create_indexes(indexes)

        current_indexes_names = list(await azure_search_manager.list_index_names())

        print(f"Количество индексов: {len(current_indexes_names)}")
        if current_indexes_names:
            print("Доступные индексы:")
            for index_name in current_indexes_names:
                print(f"  - {index_name}")
        else:
            print("Индексы не найдены.")

    finally:
        await dependencies.close()


if __name__ == '__main__':
    asyncio.run(main())
