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

    async def _hybrid_search(
            self,
            query_text: str,
            vector_fields: list[str],
            text_fields: list[str],
            top_k: int,
            semantic_config: str | None = None,
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
            semantic_config (str | None): Name of the semantic configuration to use.
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
            query_type='semantic' if semantic_config else None,
            semantic_configuration_name=semantic_config,
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
        enriched: list[dict] = [dict(document) for document in documents]

        # Process and embed text fields
        for field_to_embed in fields_to_embed:
            field_documents = [d for d in enriched if field_to_embed in d]
            texts = [d[field_to_embed] for d in field_documents]

            if texts:
                embeddings = await self._embedder.batch_embed(texts, chunk_size=16)

                for document, embedding in zip(field_documents, embeddings):
                    document[f'{field_to_embed}_vector'] = embedding

        # Upload updated documents
        await self._search_client.merge_or_upload_documents(documents=enriched)

    def _validate_batch_size(self, batch_size: int) -> None:
        """
        Validate the batch size.

        Args:
            batch_size (int): The size of the batch to validate.

        Raises:
            ValueError: If the batch size is less than or equal to 0, or greater than 1000.
        """
        if batch_size <= 0 or batch_size > 1000:
            raise ValueError(f"{self.__class__.__name__}: 'batch_size' must be between 1 and 1000")

    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the search index.

        Returns:
            int: The total number of documents in the index.
        """
        return await self._search_client.get_document_count()

    async def get_all_documents(self, batch_size: int = 500, key_field: str = 'id') -> list[dict]:
        """
        Get all document keys from the search index.

        Args:
            batch_size (int): Number of documents to retrieve per page.
            key_field (str): Field name to use as the key for deletion.

        Returns:
            list[dict]: List of documents containing their keys.
        """
        self._validate_batch_size(batch_size)

        documents = []

        results = await self._search_client.search(search_text="*", select=[key_field], top=batch_size)

        async for page in results.by_page():
            async for document in page:
                documents.append({key_field: document[key_field]})

        return documents

    async def delete_all_documents(self, batch_size: int = 500, key_field: str = 'id') -> int:
        """
        Delete all documents from the search index.

        Args:
            batch_size (int): Number of documents to delete per batch.
            key_field (str): Field name to use as the key for deletion.

        Returns:
            int: Number of documents successfully deleted.
        """
        self._validate_batch_size(batch_size)

        documents = await self.get_all_documents(batch_size=batch_size, key_field=key_field)
        deleted_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            results = await self._search_client.delete_documents(documents=batch)

            failed = [result for result in results if not result.succeeded]
            if failed:
                raise RuntimeError(f"{self.__class__.__name__}: Failed to delete {len(failed)} documents")

            deleted_count += len(results)

        return deleted_count
