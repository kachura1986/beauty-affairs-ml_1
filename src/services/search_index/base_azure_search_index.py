from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery

from ..embedding import Embedder


class BaseAzureSearchIndex:
    """Manage Azure Cognitive Search indices and perform search_index operations."""

    def __init__(self, search_client: SearchClient, embedder: Embedder) -> None:
        """
        Initializes an instance of AzureSearchIndexManager.

        Args:
            search_client (SearchClient): Client to perform asynchronous search_index operations.
            embedder (Embedder): Handles asynchronous text embedding generation.
        """
        self._search_client = search_client
        self._embedder = embedder

    async def _search(self, query_text: str, top_k: int, filters: str | None = None) -> list[dict]:
        """
        Search the search_index and retrieve the top matching results.

        Args:
            query_text (str): The search_index query text.
            top_k (int): The number of top results to retrieve.
            filters (str | None, optional): A filter expression to apply to the search_index results.

        Returns:
            list[dict]: A list of search_index results, each represented as a dictionary.
        """
        query_embedding = await self._embedder.embed(query_text)
        vector_query = VectorizedQuery(vector=query_embedding, fields='description_vector', k_nearest_neighbors=top_k)

        search_results = await self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
        )

        results_list = [result async for result in search_results]

        return results_list

    async def _delete_documents(self, documents: list[dict]) -> None:
        """
        Delete documents from the search_index search_index.

        Args:
            documents (list[dict]): A list of documents to delete.
        """
        await self._search_client.delete_documents(documents=documents)

    async def _upsert_documents(self, documents: list[dict], fields_to_embed: list[str]) -> None:
        """
        Update or upload documents to the search_index search_index.

        Args:
            documents (list[dict]): A list of documents to upload or update.
            fields_to_embed (list[dict]): A list of fields to embed.
        """
        # Process and embed text fields
        for field_to_embed in fields_to_embed:
            field_documents = [document for document in documents if field_to_embed in document]
            texts = [document[field_to_embed] for document in field_documents]

            if texts:
                embeddings = await self._embedder.batch_embed(texts, chunk_size=16)

                for document, embedding in zip(field_documents, embeddings):
                    document[f'{field_to_embed}_vector'] = embedding

        # Upload updated documents
        await self._search_client.merge_or_upload_documents(documents=documents)
