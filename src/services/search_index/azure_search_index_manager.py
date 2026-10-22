from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex


class AzureSearchIndexManager:
    """Manages Azure Cognitive Search indices and performs search_index operations."""

    def __init__(self, search_index_client: SearchIndexClient) -> None:
        """
        Initializes an instance of AzureSearchManager.

        Args:
            search_index_client (SearchIndexClient): An instance of the SearchIndexClient class.
        """
        self.search_index_client = search_index_client
        self._search_clients: dict[str, SearchClient] = {}

    async def create_indexes(self, schemas: list[SearchIndex]) -> None:
        """
        Creates search_index indexes based on the provided list of search_index schemas.

        Args:
            schemas (list[SearchIndex]): A list of search_index schemas.
        """
        for schema in schemas:
            await self.search_index_client.create_index(schema)

    async def get_index(self, index_name: str) -> SearchIndex:
        """Retrieve a search_index search_index by name.

        Args:
            index_name: The name of the search_index.

        Returns:
            SearchIndex: The search_index schema retrieved from Azure AI Search.
        """
        return await self.search_index_client.get_index(index_name)

    def get_instance_index(self, index_name: str):
        """
        Retrieve a search_index client for a given search_index name.

        Args:
            index_name (str): The name of the search_index.

        Returns:
            SearchClient: The search_index client for the given search_index name.
        """
        if index_name not in self._search_clients:
            self._search_clients[index_name] = self.search_index_client.get_search_client(index_name)

        return self._search_clients[index_name]

    async def list_index_names(self) -> set[str]:
        """
        Lists all search_index names available in the search_index service.

        Returns:
            set[str]: A set of all search_index names.
        """
        return {index_name async for index_name in self.search_index_client.list_index_names()}

    async def delete_indexes(self, index_names: list[str]) -> None:
        """
        Deletes search_index indexes based on the provided list of search_index names.

        Args:
            index_names (list[str]): A list of search_index names to delete.
        """
        for index_name in index_names:
            await self.search_index_client.delete_index(index_name)

    async def close(self) -> None:
        """Closes all search_index clients."""
        try:
            for client in self._search_clients.values():
                await client.close()
        finally:
            self._search_clients.clear()
            await self.search_index_client.close()
