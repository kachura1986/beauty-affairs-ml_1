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
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)

from core.config import settings
from .azure_search_index_manager import AzureSearchIndexManager
from .base_azure_search_index import BaseAzureSearchIndex
from ..embedding import Embedder


class ProductsSearchIndex(BaseAzureSearchIndex):
    _vector_fields = ['name_vector', 'description_vector']
    _text_fields = ['name', 'category_name', 'subcategory_name', 'description', 'brand_name']
    _semantic_config = 'products-semantic-config'

    @classmethod
    def build_schema(cls, index_name: str) -> SearchIndex:
        """
        Build the product search_index search_index schema.

        Args:
            index_name (str): The name of the search_index.

        Returns:
            SearchIndex: The created product search_index search_index schema.
        """
        # Define search_index fields
        fields = [
            SimpleField(name='id', type='Edm.String', key=True),
            SimpleField(name='price', type='Edm.Double', filterable=True),
            SimpleField(name='quantity_value', type='Edm.Double'),
            SimpleField(name='quantity_unit', type='Edm.String'),
            # Text search_index fields
            SearchableField(name='name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='category_name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='subcategory_name', type='Edm.String', analyzer_name='en.lucene'),
            SearchableField(name='description', type='Edm.String', analyzer_name='en.lucene', filterable=True),
            SearchableField(name='brand_name', type='Edm.String', analyzer_name='en.lucene', filterable=True),
            # Vector search_index fields (for new vector field given 'field name_vector')
            SearchField(
                name='name_vector',
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=settings.EMBEDDING_DIMENSIONS,
                vector_search_profile_name='HNSWProfile'
            ),
            SearchField(
                name='description_vector',
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=settings.EMBEDDING_DIMENSIONS,
                vector_search_profile_name='HNSWProfile'
            ),
        ]

        # Define vector search_index configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name='HNSW',
                    parameters=HnswParameters(
                        metric='cosine'
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name='HNSWProfile',
                    algorithm_configuration_name='HNSW'
                )
            ]
        )

        # Define semantic search_index configuration
        semantic_search = SemanticSearch(
            default_configuration_name='products-semantic-config',
            configurations=[
                SemanticConfiguration(
                    name='products-semantic-config',
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name='name'),
                        content_fields=[SemanticField(field_name='description')],
                        keywords_fields=[
                            SemanticField(field_name='category_name'),
                            SemanticField(field_name='subcategory_name'),
                            SemanticField(field_name='brand_name'),
                        ],
                    ),
                )
            ],
        )

        schema = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        return schema

    def __init__(self, index_name: str, azure_search_index_manager: AzureSearchIndexManager, embedder: Embedder):
        super().__init__(search_client=azure_search_index_manager.get_instance_index(index_name), embedder=embedder)

        self._fields_to_embed = ['name', 'description']
