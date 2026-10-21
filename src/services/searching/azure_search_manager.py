import asyncio
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

from .base_azure_search_index import BaseAzureSearchIndex
from ..embedding.azure_openai_embedder import AzureOpenAIEmbedder
from .products_index import ProductsSearchIndex
from .services_index import ServicesSearchIndex


class AzureSearchManager:
    """
    Manages Azure Cognitive Search indices and performs search operations.

    Attributes:
        search_endpoint (str): The endpoint of the Azure Cognitive Search service.
        credentials (AzureKeyCredential): Credentials for accessing the Azure Cognitive Search service.
        search_index_client (SearchIndexClient): An instance of the SearchIndexClient class.
        embeddings_client (AzureOpenAIEmbedder): An instance of the AzureOpenAIEmbedder class.
        index_classes (dict): A dictionary mapping index types to corresponding index classes.
    """

    def __init__(self) -> None:
        """
        Initializes an instance of AzureSearchManager.
        """
        # Service credentials and client initialization.
        self.search_endpoint = os.getenv("SEARCH_ENDPOINT")
        self.credentials = AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
        self.search_index_client = SearchIndexClient(
            self.search_endpoint, self.credentials
        )

        # Additional clients.
        self.embeddings_client = AzureOpenAIEmbedder()

        # Index type mapping.
        self.index_classes = {
            "products": ProductsSearchIndex,
            "services": ServicesSearchIndex,
        }

    def _get_index_instance(
            self, index_type: str, index_name: str
    ) -> BaseAzureSearchIndex | None:
        """
        Retrieves an instance of a search index based on its type and name.

        Args:
            index_type (str): The type of the search index ('products' or 'services').
            index_name (str): The name of the search index.

        Returns:
            BaseAzureSearchIndex | None: An instance of the corresponding index class, or None if not found.
        """
        index_class = self.index_classes.get(index_type)
        if not index_class:
            return None

        search_client = SearchClient(self.search_endpoint, index_name, self.credentials)
        index_instance = index_class(
            index_name, search_client, self.embeddings_client
        )

        return index_instance

    def get_index_instance(
            self, index_type: str, index_name: str
    ) -> BaseAzureSearchIndex | None:
        return self._get_index_instance(index_type, index_name)

    async def list_index_names(self) -> set[str]:
        return {
            index_name
            async for index_name in self.search_index_client.list_index_names()
        }

    async def _create_indexes(self, indexes: dict[str, list[str]]) -> None:
        """
        Creates search indexes based on the provided dictionary of index names.

        Args:
            indexes (dict[str, list[str]]): A dictionary where the key is the index type ('products' or 'services') and
            the value is a list of index names to be created.
        """
        for index_type, index_names in indexes.items():
            for index_name in index_names:
                index_instance = self._get_index_instance(index_type, index_name)
                if index_instance:
                    schema = index_instance.build_schema()
                    await self.search_index_client.create_index(schema)

    async def _update_documents(
            self, all_documents: dict[str, dict[str, list[dict]]]
    ) -> None:
        """
        Updates or uploads product and service documents to their respective search indexes.

        Args:
            all_documents (dict): A dictionary containing 'products' and 'services' documents.
        """
        for index_type, documents in all_documents.items():
            for index_name, docs in documents.items():
                index_instance = self._get_index_instance(index_type, index_name)

                if index_instance:
                    await index_instance.update_documents(docs)

    async def _delete_indexes(self, index_names: list[str]) -> None:
        """
        Deletes search indexes based on the provided list of index names.

        Args:
            index_names (list[str]): A list of index names to delete.
        """
        for index_name in index_names:
            await self.search_index_client.delete_index(index_name)

    async def create_indexes(
            self, products: dict[str, list[dict]], services: dict[str, list[dict]]
    ) -> None:
        """
        Creates or updates search indexes based on provided products and services.

        Args:
            products (dict): Product documents to be indexed.
            services (dict): Service documents to be indexed.
        """
        # Index names.
        product_indexes = set(products.keys())
        service_indexes = set(services.keys())

        # Delete existing indexes.
        existing_indexes = await self.list_index_names()
        new_indexes = product_indexes | service_indexes
        indexes_to_delete = existing_indexes & new_indexes

        if indexes_to_delete:
            await self._delete_indexes(list(indexes_to_delete))

        # Create new indexes.
        indexes_to_create = {
            "products": list(product_indexes),
            "services": list(service_indexes),
        }
        await self._create_indexes(indexes_to_create)

        # Update documents.
        await self._update_documents({"products": products, "services": services})

    async def update_documents(
            self, products: dict[str, list[dict]], services: dict[str, list[dict]]
    ) -> None:
        """
        Updates product and service documents in their respective search indexes.

        Args:
            products (dict): Product documents to update.
            services (dict): Service documents to update.
        """
        # Index names
        product_indexes = set(products.keys())
        service_indexes = set(services.keys())

        existing_indexes = await self.list_index_names()

        indexes_to_create = {
            "products": list(product_indexes - existing_indexes),
            "services": list(service_indexes - existing_indexes),
        }

        # Create indexes
        await self._create_indexes(indexes_to_create)

        # Update documents
        await self._update_documents({"products": products, "services": services})

    async def delete_documents(
            self, products: dict[str, list[dict]], services: dict[str, list[dict]]
    ) -> None:
        """
        Deletes product and service documents from search indexes.

        Args:
            products (dict): Product documents to delete.
            services (dict): Service documents to delete.
        """
        existing_indexes = await self.list_index_names()

        for index_type, documents in {
            "products": products,
            "services": services,
        }.items():
            # Delete documents.
            for index_name, docs in documents.items():
                # Skip if index does not exist.
                if index_name not in existing_indexes:
                    continue

                index_instance = self._get_index_instance(index_type, index_name)

                if index_instance:
                    await index_instance.delete_documents(docs)

                    # Ensure index is empty before deletion.
                    for _ in range(3):
                        if await index_instance._search_client.get_document_count() == 0:
                            await self.search_index_client.delete_index(index_name)
                            # Update set to avoid rechecking
                            existing_indexes.discard(index_name)
                            break
                        await asyncio.sleep(1)
