from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from .azure_search_index_manager import AzureSearchIndexManager
from .base_azure_search_index import BaseAzureSearchIndex
from ..embedding import Embedder


class ServicesSearchIndex(BaseAzureSearchIndex):
    @classmethod
    def build_schema(cls, index_name: str) -> SearchIndex:
        """
        Build the service search_index search_index schema.

        Args:
            index_name (str): The name of the search_index.

        Returns:
            SearchIndex: The created service search_index search_index schema.
        """
        # Define search_index fields
        fields = [
            SimpleField(name='id', type='Edm.String', key=True),
            SimpleField(name='price', type='Edm.Double'),
            SimpleField(name='duration', type='Edm.Int32'),
            SimpleField(name='recovery_time', type='Edm.Int32'),
            # Text search_index fields
            SearchableField(name='name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='category_name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='subcategory_name', type='Edm.String', analyzer_name='en.lucene'),
            # Filterable search_index fields
            SearchableField(name='description', type='Edm.String', analyzer_name='en.lucene', filterable=True),
            # Vector search_index fields (for new vector field given 'field name_vector')
            SearchField(
                name='name_vector',
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1024,
                vector_search_profile_name='HNSWProfile'
            ),
            SearchField(
                name='description_vector',
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1024,
                vector_search_profile_name='HNSWProfile'
            )
        ]

        # Define vector search_index configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name='HNSW',
                    parameters=HnswParameters(metric='cosine')
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name='HNSWProfile',
                    algorithm_configuration_name='HNSW'
                )
            ]
        )

        schema = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

        return schema

    def __init__(self, index_name: str, azure_search_index_manager: AzureSearchIndexManager, embedder: Embedder):
        self._search_client = azure_search_index_manager.get_instance_index(index_name)
        self._embedder = embedder
        super().__init__(self._search_client, self._embedder)

        self._fields_to_embed = ['name', 'description']

    async def upsert_documents(self, documents: list[dict]) -> None:
        """
        Update or upload documents to the search_index search_index.

        Args:
            documents (list[dict]): A list of documents to upload or update.
        """
        await self._upsert_documents(documents=documents, fields_to_embed=self._fields_to_embed)

    async def delete_documents(self, documents: list[dict]) -> None:
        """
        Delete documents from the search_index search_index.

        Args:
            documents (list[dict]): A list of documents to delete.
        """
        await self._search_client.delete_documents(documents=documents)
