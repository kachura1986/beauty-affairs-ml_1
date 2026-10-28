from services.embedding import Embedder
from services.search_index import AzureSearchIndexManager, ProductsSearchIndex, ServicesSearchIndex


class IndexService:
    """Management of search_index indexes for products and services."""

    def __init__(self, search_index_manager: AzureSearchIndexManager, embedder: Embedder) -> None:
        self._search_index_manager = search_index_manager
        self._embedder = embedder

        # Index type mapping
        self.index_classes = {
            'products': ProductsSearchIndex,
            'services': ServicesSearchIndex,
        }

    async def _create_indexes(self, indexes: dict[str, list[str]]) -> None:
        """
        Creates search_index indexes based on the provided dictionary of search_index names.

        Args:
            indexes (dict[str, list[str]]): A dictionary where the key is the type of search_index search_index
            ('products' or 'services') and the value is a list of search_index names to be created.
        """
        for index_type, index_names in indexes.items():
            index_class = self.index_classes.get(index_type)
            if index_class is None:
                raise ValueError(f"{self.__class__.__name__}: Unsupported index type: {index_type}")

            schemas = []
            for index_name in index_names:
                schema = index_class.build_schema(index_name)
                schemas.append(schema)

            if schemas:
                await self._search_index_manager.create_indexes(schemas)

    async def _upsert_documents(self, all_documents: dict[str, dict[str, list[dict]]]) -> None:
        """
        Updates or uploads product and service documents to their respective search_index indexes.

        Args:
            all_documents (dict[str, dict[str, list[dict]]]): A dictionary containing 'products' and
            'services' documents.
        """
        for index_type, documents in all_documents.items():
            index_class = self.index_classes.get(index_type)
            if index_class is None:
                raise ValueError(f"{self.__class__.__name__}: Unsupported index type: {index_type}")

            for index_name, docs in documents.items():
                if not docs:
                    continue

                index_instance = index_class(
                    index_name=index_name,
                    azure_search_index_manager=self._search_index_manager,
                    embedder=self._embedder,
                )

                await index_instance.upsert_documents(docs)

    async def upsert_documents(self, products: dict[str, list[dict]], services: dict[str, list[dict]]) -> None:
        """
        Updates product and service documents in their respective search_index indexes.

        Args:
            products dict[str, list[dict]]: Product documents to update.
            services dict[str, list[dict]]: Service documents to update.
        """
        # Index names
        product_indexes = set(products.keys())
        service_indexes = set(services.keys())

        conflicting_indexes = product_indexes & service_indexes
        if conflicting_indexes:
            raise ValueError(
                f"{self.__class__.__name__}: Index names cannot be shared by products and services: "
                f"{sorted(conflicting_indexes)}"
            )

        existing_indexes = await self._search_index_manager.list_index_names()

        indexes_to_create = {
            'products': list(product_indexes - existing_indexes),
            'services': list(service_indexes - existing_indexes),
        }

        # Create indexes
        if any(indexes_to_create.values()):
            await self._create_indexes(indexes_to_create)

        # Update documents
        await self._upsert_documents({'products': products, 'services': services})

    async def _delete_documents(self, all_documents: dict[str, dict[str, list[dict]]]) -> None:
        """
        Deletes product and service documents from their respective indexes.

        Args:
            all_documents (dict[str, dict[str, list[dict]]]): A dictionary containing 'products' and 'services'
            documents grouped by index name.
        """
        for index_type, documents in all_documents.items():
            index_class = self.index_classes.get(index_type)
            if index_class is None:
                raise ValueError(f"{self.__class__.__name__}: Unsupported index type: {index_type}")

            for index_name, docs in documents.items():
                if not docs:
                    continue

                index_instance = index_class(
                    index_name=index_name,
                    azure_search_index_manager=self._search_index_manager,
                    embedder=self._embedder,
                )

                await index_instance.delete_documents(docs)

    async def delete_documents(self, products: dict[str, list[dict]], services: dict[str, list[dict]]) -> None:
        """
        Deletes product and service documents from search index indexes.

        Args:
            products dict[str, list[dict]]: Product documents grouped by index name.
            services dict[str, list[dict]]: Service documents grouped by index name.
        """
        product_indexes = set(products.keys())
        service_indexes = set(services.keys())

        conflicting_indexes = product_indexes & service_indexes
        if conflicting_indexes:
            raise ValueError(
                f"{self.__class__.__name__}: Index names cannot be shared by products and services: "
                f"{sorted(conflicting_indexes)}"
            )

        await self._delete_documents({'products': products, 'services': services})
