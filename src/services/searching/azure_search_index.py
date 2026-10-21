from typing import Protocol

from azure.search.documents.indexes.models import SearchIndex


class AzureSearchIndex(Protocol):
    """Define the application-level interface for asynchronous Azure Search indices."""

    def build_schema(self) -> SearchIndex:
        """
        Build the search index schema.

        Returns:
            SearchIndex: The created search index schema.
        """
        ...

    async def search(self, query_text: str, top_k: int, filters: str | None = None) -> list[dict]:
        """
        Search the index and retrieve the top matching results.

        Args:
            query_text (str): The search query text.
            top_k (int): The number of top results to retrieve.
            filters (str | None, optional): A filter expression to apply to the search results.

        Returns:
            list[dict]: A list of search results, each represented as a dictionary.
        """
        ...

    async def delete_documents(self, documents: list[dict]) -> None:
        """
        Delete documents from the search index.

        Args:
            documents (list[dict]): A list of documents to delete.
        """
        ...

    async def update_documents(self, documents: list[dict]) -> None:
        """
        Update or upload documents to the search index.

        Args:
            documents (list[dict]): A list of documents to upload or update.
        """
        ...
