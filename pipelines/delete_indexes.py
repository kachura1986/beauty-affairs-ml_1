import asyncio

from dependencies import PipelineDependencies


async def main(index_names: list[str]) -> None:
    dependencies = await PipelineDependencies.create()
    azure_search_manager = dependencies.azure_search_index_manager

    try:
        print("Deleting indexes")
        await azure_search_manager.delete_indexes(index_names)

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
    index_names = []

    asyncio.run(main(index_names))
