import asyncio
import logging

from dependencies import PipelineDependencies

logger = logging.getLogger(__name__)


async def main(index_names: list[str]) -> None:
    dependencies = await PipelineDependencies.create()

    try:
        print("Deleting indexes")
        await dependencies.azure_search_manager._delete_indexes(index_names)

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

    index_names = []
    asyncio.run(main(index_names))
