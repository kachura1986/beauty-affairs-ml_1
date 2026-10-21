import asyncio

from .azure_search_manager import AzureSearchManager


class SearchService:
    """Performs search queries for products and services."""

    def __init__(self, search_manager: AzureSearchManager) -> None:
        self._search_manager = search_manager

    async def _search(self, query: str, index_name: str, index_type: str, filters: str | None = None) -> list[dict]:
        """
        Performs a search query on the specified index.

        Args:
            query (str): The search query.
            index_name (str): The name of the index to search in.
            index_type (str): The type of the index to search in.
            filters (str | None): Optional filters to apply to the search query.

        Returns:
            list[dict]: A list of search results.
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
