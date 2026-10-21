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

from .base_azure_search_index import BaseAzureSearchIndex


class ServicesSearchIndex(BaseAzureSearchIndex):
    """Define the schema for service search indices."""

    def build_schema(self) -> SearchIndex:
        """
        Build the service search index schema.

        Returns:
            SearchIndex: The created service search index schema.
        """
        # Define search fields
        fields = [
            SimpleField(name='id', type='Edm.String', key=True),
            SimpleField(name='price', type='Edm.Double'),
            SimpleField(name='duration', type='Edm.Int32'),
            SimpleField(name='recovery_time', type='Edm.Int32'),
            # Text search fields
            SearchableField(name='name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='category_name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='subcategory_name', type='Edm.String', analyzer_name='en.lucene'),
            # Filterable search fields
            SearchableField(name='description', type='Edm.String', analyzer_name='en.lucene', filterable=True),
            # Vector search fields
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

        # Define vector search configuration
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

        schema = SearchIndex(name=self._index_name, fields=fields, vector_search=vector_search)

        return schema
