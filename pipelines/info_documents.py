import asyncio

from dependencies import PipelineDependencies


async def main() -> None:
    """
    Creates product and service indexes.
    """
    dependencies = await PipelineDependencies.create()
    azure_search_manager = dependencies.azure_search_index_manager

    try:
        # Indexes info
        print("Indexes info")
        index_names = await azure_search_manager.list_index_names()

        search_indexes = [azure_search_manager.get_instance_index(index_name) for index_name in index_names]
        for index_name, search_index in zip(index_names, search_indexes):
            document_count = await search_index.get_document_count()
            print(f"{index_name}: {document_count}")

    finally:
        await dependencies.close()


if __name__ == '__main__':
    asyncio.run(main())
