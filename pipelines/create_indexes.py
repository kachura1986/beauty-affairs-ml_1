import asyncio
import logging

from dependencies import PipelineDependencies

logger = logging.getLogger(__name__)


async def main(indexes: dict[str, list[str]]) -> None:
    """
    Creates product and service indexes.

    Args:
        indexes (dict[str, list[str]]): A dictionary where the key is the type of search index
        ('products' or 'services') and the value is a list of index names to be created.
    """
    dependencies = await PipelineDependencies.create()

    try:
        print("Creating indexes")
        await dependencies.azure_search_manager._create_indexes(indexes)

        current_indexes_names = list(await dependencies.azure_search_manager.list_index_names())

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
    logging.basicConfig(level=logging.INFO)

    indexes = {
        'products': ['products_test'],
        'services': ['services_test'],
    }
    asyncio.run(main(indexes))
