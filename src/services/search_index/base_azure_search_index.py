from azure.search.documents.aio import SearchClient
from azure.search.documents.models import HybridSearch, VectorizedQuery

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

    async def _hybrid_search(
            self,
            query_text: str,
            vector_fields: list[str],
            text_fields: list[str],
            top_k: int,
            vector_candidates: int | None = None,
            filters: str | None = None,
    ) -> list[dict]:
        """
        Perform hybrid text and vector search.

        Args:
            query_text (str): Text used for both text search and embedding generation.
            vector_fields (list[str]): Names of vector fields to search.
            text_fields (list[str]): Names of searchable text fields.
            top_k (int): Number of documents to return.
            vector_candidates (int | None): Number of candidate documents to consider for vector search.
                If None, defaults to top_k.
            filters (str | None): OData filter expression. If None, no filter is applied.

        Returns:
            list[dict]: A list of matching documents.
        """
        # Generate the query embedding once
        query_embedding = await self._embedder.embed(query_text)

        # Search each configured vector field
        vector_query = VectorizedQuery(
            vector=query_embedding,
            fields=', '.join(vector_fields),
            k_nearest_neighbors=vector_candidates if vector_candidates is not None else top_k,
        )

        # Combine full-text search and vector search
        search_results = await self._search_client.search(
            search_text=query_text if text_fields else None,
            search_fields=text_fields,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
        )

        return [result async for result in search_results]

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

    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the search index.

        Returns:
            int: The total number of documents in the index.
        """
        return await self._search_client.get_document_count()
