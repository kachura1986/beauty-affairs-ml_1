from typing import Protocol


class AzureSearchIndex(Protocol):
    """Define the application-level interface for asynchronous Azure Search indices."""

    async def search(self, query_text: str, top_k: int, filters: str | None = None) -> list[dict]:
        """
        Search the search_index and retrieve the top matching results.

        Args:
            query_text (str): The search_index query text.
            top_k (int): The number of top results to retrieve.
            filters (str | None, optional): A filter expression to apply to the search_index results.

        Returns:
            list[dict]: A list of search_index results, each represented as a dictionary.
        """
        ...

    async def delete_documents(self, documents: list[dict]) -> None:
        """
        Delete documents from the search_index search_index.

        Args:
            documents (list[dict]): A list of documents to delete.
        """
        ...

    async def update_documents(self, documents: list[dict]) -> None:
        """
        Update or upload documents to the search_index search_index.

        Args:
            documents (list[dict]): A list of documents to upload or update.
        """
        ...
