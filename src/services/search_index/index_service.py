import asyncio

from azure.search.documents.aio import SearchClient

from .azure_index_manager import AzureSearchManager
from .products_index import ProductsSearchIndex
from .services_index import ServicesSearchIndex


class IndexService:
    """Performs search_index queries for products and services."""

    def __init__(self, search_manager: AzureSearchManager) -> None:
        self._search_manager = search_manager

        self._search_clients: dict[str, SearchClient] = {}

        # Index type mapping
        self.index_classes = {
            'products': ProductsSearchIndex,
            'services': ServicesSearchIndex,
        }


    def get_index_instance(self, index_type: str, index_name: str) -> BaseAzureSearchIndex | None:
        """
        Retrieves an instance of a search_index search_index based on its type and name.

        Args:
            index_type (str): The type of the search_index search_index ('products' or 'services').
            index_name (str): The name of the search_index search_index.

        Returns:
            BaseAzureSearchIndex | None: An instance of the corresponding search_index class, or None if not found.
        """
        index_class = self.index_classes.get(index_type)
        if not index_class:
            return None

        search_client = self._get_search_client(index_name)
        index_instance = index_class(index_name, search_client, self.embeddings_client)

        return index_instance

    def _get_search_client(self, index_name: str) -> SearchClient:
        """
        Retrieve a search_index client for a given search_index name.

        Args:
            index_name (str): The name of the search_index.

        Returns:
            SearchClient: The search_index client for the given search_index name.
        """
        search_client = self._search_clients.get(index_name)

        if search_client is None:
            search_client = self.search_index_client.get_search_client(index_name)
            self._search_clients[index_name] = search_client

        return search_client

    async def _search(self, query: str, index_name: str, index_type: str, filters: str | None = None) -> list[dict]:
        """
        Performs a search_index query on the specified search_index.

        Args:
            query (str): The search_index query.
            index_name (str): The name of the search_index to search_index in.
            index_type (str): The type of the search_index to search_index in.
            filters (str | None): Optional filters to apply to the search_index query.

        Returns:
            list[dict]: A list of search_index results.
        """
        index_instance = self._search_manager.get_index_instance(index_type, index_name)
        results = (
            await index_instance.search(query_text=query, top_k=50, filters=filters)
            if index_instance
            else []
        )

        return results

    def _create_brand_filter(self, brand_list: list[str], exclude: bool) -> str:
        """
        Creates a string representing a filter for brands.

        Args:
            brand_list (list[str]): List of brand names.
            exclude (bool): Whether to exclude or include the brands.

        Returns:
            str: Filter for brands.
        """
        # Replace single quotes in brand names to prevent syntax errors in the filter
        query = ' or '.join([f"brand_name eq '{brand.replace(chr(39), chr(39) * 2)}'" for brand in brand_list])

        if exclude:
            return f'not ({query})'

        return f'({query})'

    async def search(
            self,
            products_recommends: list[str],
            services_recommends: list[str],
            center_id: str,
            brand_list: list[str],
            exclude: bool,
    ) -> tuple[list[list[dict]], list[list[dict]]]:
        """
        Searches for products and services based on recommendations.

        Args:
            products_recommends (list[str]): List of recommended products.
            services_recommends (list[str]): List of recommended services.
            center_id (str): Center ID.
            brand_list (list[str]): List of brand names.
            exclude (bool): Whether to exclude or include the brands.

        Returns:
            tuple[dict[str, list[dict]], dict[str, list[dict]]]: Search results for products and services.
        """
        existing_indexes = await self._search_manager.list_index_names()

        brand_filters = self._create_brand_filter(brand_list, exclude)
        description_filters = "description ne ''"

        async def search_product(product_recommend):
            index_name = f'products_{center_id}'
            filters_str = f'{description_filters}'
            if brand_list:
                filters_str += f' and {brand_filters}'

            return (
                await self._search(product_recommend, index_name, 'products', filters=filters_str)
                if index_name in existing_indexes
                else []
            )

        async def search_service(service_recommend):
            index_name = f'services_{center_id}'
            return (
                await self._search(service_recommend, index_name, 'services', filters=description_filters)
                if index_name in existing_indexes
                else []
            )

        product_results = await asyncio.gather(
            *(search_product(product_recommend) for product_recommend in products_recommends)
        )
        service_results = await asyncio.gather(
            *(search_service(service_recommend) for service_recommend in services_recommends)
        )

        return list(product_results), list(service_results)

    async def _create_indexes(self, indexes: dict[str, list[str]]) -> None:
        """
        Creates search_index indexes based on the provided dictionary of search_index names.

        Args:
            indexes (dict[str, list[str]]): A dictionary where the key is the type of search_index search_index
            ('products' or 'services') and the value is a list of search_index names to be created.
        """
        for index_type, index_names in indexes.items():
            for index_name in index_names:
                index_instance = self._get_index_instance(index_type, index_name)
                if index_instance:
                    schema = index_instance.build_schema()
                    await self.search_index_client.create_index(schema)

    async def _update_documents(self, all_documents: dict[str, dict[str, list[dict]]]) -> None:
        """
        Updates or uploads product and service documents to their respective search_index indexes.

        Args:
            all_documents (dict): A dictionary containing 'products' and 'services' documents.
        """
        for index_type, documents in all_documents.items():
            for index_name, docs in documents.items():
                index_instance = self._get_index_instance(index_type, index_name)

                if index_instance:
                    await index_instance.upsert_documents(docs)

    async def create_indexes(self, products: dict[str, list[dict]], services: dict[str, list[dict]]) -> None:
        """
        Creates or updates search_index indexes based on provided products and services.

        Args:
            products (dict): Product documents to be indexed.
            services (dict): Service documents to be indexed.
        """
        # Index names
        product_indexes = set(products.keys())
        service_indexes = set(services.keys())

        # Delete existing indexes
        existing_indexes = await self.list_index_names()
        new_indexes = product_indexes | service_indexes
        indexes_to_delete = existing_indexes & new_indexes

        if indexes_to_delete:
            await self._delete_indexes(list(indexes_to_delete))

        # Create new indexes
        indexes_to_create = {
            'products': list(product_indexes),
            'services': list(service_indexes),
        }
        await self._create_indexes(indexes_to_create)

        # Update documents
        await self._update_documents({'products': products, 'services': services})

    async def update_documents(self, products: dict[str, list[dict]], services: dict[str, list[dict]]) -> None:
        """
        Updates product and service documents in their respective search_index indexes.

        Args:
            products (dict): Product documents to update.
            services (dict): Service documents to update.
        """
        # Index names
        product_indexes = set(products.keys())
        service_indexes = set(services.keys())

        existing_indexes = await self.list_index_names()

        indexes_to_create = {
            'products': list(product_indexes - existing_indexes),
            'services': list(service_indexes - existing_indexes),
        }

        # Create indexes
        await self._create_indexes(indexes_to_create)

        # Update documents
        await self._update_documents({'products': products, 'services': services})

    async def delete_documents(self, products: dict[str, list[dict]], services: dict[str, list[dict]]) -> None:
        """
        Deletes product and service documents from search_index indexes.

        Args:
            products (dict): Product documents to delete.
            services (dict): Service documents to delete.
        """
        existing_indexes = await self.list_index_names()

        for index_type, documents in {
            'products': products,
            'services': services,
        }.items():
            # Delete documents
            for index_name, docs in documents.items():
                # Skip if search_index does not exist
                if index_name not in existing_indexes:
                    continue

                index_instance = self._get_index_instance(index_type, index_name)

                if index_instance:
                    await index_instance.delete_documents(docs)

                    # Ensure search_index is empty before deletion
                    for _ in range(3):
                        if await index_instance._search_client.get_document_count() == 0:
                            await self.search_index_client.delete_index(index_name)
                            # Update set to avoid rechecking
                            existing_indexes.discard(index_name)
                            break
                        await asyncio.sleep(1)


    async def close(self) -> None:
        """Closes all search_index clients."""
        for search_client in self._search_clients.values():
            await search_client.close()

        self._search_clients.clear()
