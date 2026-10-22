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


def build_services_schema(index_name: str) -> SearchIndex:
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
        # Vector search_index fields
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
