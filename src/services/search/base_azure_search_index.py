from abc import ABC, abstractmethod

from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import VectorizedQuery

from .azure_search_index import AzureSearchIndex
from ..embedding import Embedder


class BaseAzureSearchIndex(AzureSearchIndex, ABC):
    """Define the shared implementation for asynchronous Azure Search indices."""

    def __init__(self, index_name: str, search_client: SearchClient, embedder: Embedder) -> None:
        """
        Initialize the search index with a search client and an embedder.

        Args:
            index_name (str): The name of the search index.
            search_client (SearchClient): Client to perform asynchronous search operations.
            embedder (Embedder): Handles asynchronous text embedding generation.
        """
        self._index_name = index_name
        self._search_client = search_client
        self._embedder = embedder

    @abstractmethod
    def build_schema(self) -> SearchIndex:
        """
        Build the search index schema.

        Returns:
            SearchIndex: The created search index schema.
        """
        pass

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

    async def delete_documents(self, documents: list[dict]) -> None:
        """
        Delete documents from the search index.

        Args:
            documents (list[dict]): A list of documents to delete.
        """
        await self._search_client.delete_documents(documents=documents)

    async def update_documents(self, documents: list[dict]) -> None:
        """
        Update or upload documents to the search index.

        Args:
            documents (list[dict]): A list of documents to upload or update.
        """
        vector_fields = ['name', 'description']

        # Process and embed text fields
        for field in vector_fields:
            field_documents = [document for document in documents if field in document]
            texts = [document[field] for document in field_documents]

            if texts:
                embeddings = await self._embedder.batch_embed(texts, chunk_size=16)

                for document, embedding in zip(field_documents, embeddings):
                    document[f'{field}_vector'] = embedding

        # Upload updated documents
        await self._search_client.merge_or_upload_documents(documents=documents)
