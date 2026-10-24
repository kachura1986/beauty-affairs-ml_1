from pydantic import BaseModel, Field

from .registry import ToolRegistry
from .tool import Tool
from ...search_index.products_search_index import ProductsSearchIndex


class SearchProductsArgs(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Search query describing the product. "
            "Can include product name, brand, category, or desired characteristics."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of products to return.",
    )
    brand_name: str | None = Field(
        default=None,
        description="Optional brand filter.",
    )
    category_name: str | None = Field(
        default=None,
        description="Optional product category filter.",
    )
    min_price: float | None = Field(
        default=None,
        ge=0,
        description="Minimum product price.",
    )
    max_price: float | None = Field(
        default=None,
        ge=0,
        description="Maximum product price.",
    )


def _odata_escape(value: str) -> str:
    """Escape a string for an OData string literal."""
    return value.replace("'", "''")


def _build_product_filter(args: SearchProductsArgs) -> str | None:
    conditions = []

    if args.brand_name:
        value = _odata_escape(args.brand_name)
        conditions.append(f"brand_name eq '{value}'")

    if args.category_name:
        value = _odata_escape(args.category_name)
        conditions.append(f"category_name eq '{value}'")

    if args.min_price is not None:
        conditions.append(f"price ge {args.min_price}")

    if args.max_price is not None:
        conditions.append(f"price le {args.max_price}")

    return " and ".join(conditions) if conditions else None


def create_product_tools(products_index: ProductsSearchIndex) -> ToolRegistry:
    async def search_products(
            query: str,
            top_k: int = 10,
            brand_name: str | None = None,
            category_name: str | None = None,
            min_price: float | None = None,
            max_price: float | None = None,
    ) -> list[dict]:
        args = SearchProductsArgs(
            query=query,
            top_k=top_k,
            brand_name=brand_name,
            category_name=category_name,
            min_price=min_price,
            max_price=max_price,
        )

        if (
                args.min_price is not None
                and args.max_price is not None
                and args.min_price > args.max_price
        ):
            raise ValueError("min_price must be less than or equal to max_price")

        filters = _build_product_filter(args)

        documents = await products_index.search_documents(
            query_text=args.query,
            top_k=args.top_k,
            filters=filters,
        )

        # Return only fields useful to the model
        fields = (
            "id",
            "name",
            "description",
            "brand_name",
            "category_name",
            "subcategory_name",
            "price",
            "quantity_value",
            "quantity_unit",
        )

        return [
            {key: document[key] for key in fields if key in document}
            for document in documents
        ]

    return ToolRegistry([
        Tool(
            name="Searching products",
            description=(
                "Search the product catalog using hybrid text and vector search. Use this tool when the user asks "
                "to find, compare, or recommend products available in the catalog. Supports optional brand, "
                "category, and price filters. Returns matching product details. Do not invent products "
                "or prices that are not present in the results."
            ),
            fn=search_products,
            args_model=SearchProductsArgs,
        ),
    ])
