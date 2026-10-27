from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import ToolRegistry
from .tool import Tool
from ...search_index.products_search_index import ProductsSearchIndex


class SearchProductsArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    query: str = Field(
        min_length=1,
        max_length=500,
        description="""
Free-text search query describing the products to find. Can include product names, brand names, product features, 
intended benefits, usage instructions, ingredients and desired characteristics to improve text and vector search relevance. 
Structured price constraints must be provided through their dedicated fields;
mentioning them in the query alone does not apply structured filters.
""",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of products to return. Must be between 1 and 50. Defaults to 10.",
    )
    min_price: float | None = Field(
        default=None,
        ge=0,
        description="""
Optional inclusive minimum product price. When provided, adds the OData condition price ge <min_price>, 
requiring the product price to be greater than or equal to this value. Must be non-negative. No minimum-price
condition is applied when this field is omitted.
""",
    )
    max_price: float | None = Field(
        default=None,
        ge=0,
        description="""
Optional inclusive maximum product price. When provided, adds the OData condition price le <max_price>, 
requiring the product price to be less than or equal to this value. Must be non-negative. No maximum-price condition is 
applied when this field is omitted.
""",
    )

    @model_validator(mode="after")
    def validate_price_range(self) -> "SearchProductsArgs":
        if (
                self.min_price is not None
                and self.max_price is not None
                and self.min_price > self.max_price
        ):
            raise ValueError("min_price must be less than or equal to max_price")

        return self


def _build_product_filter(min_price: float | None, max_price: float | None) -> str | None:
    conditions = []

    if min_price is not None:
        conditions.append(f"price ge {min_price}")

    if max_price is not None:
        conditions.append(f"price le {max_price}")

    return " and ".join(conditions) if conditions else None


def create_product_tools(products_index: ProductsSearchIndex) -> ToolRegistry:
    async def search_products(
            query: str,
            top_k: int = 10,
            min_price: float | None = None,
            max_price: float | None = None,
    ) -> list[dict]:
        filters = _build_product_filter(min_price, max_price)

        documents = await products_index.search_documents(query_text=query, top_k=top_k, filters=filters)

        # Return only fields useful to the model
        fields = (
            'id',
            'name',
            'description',
            'brand_name',
            'category_name',
            'subcategory_name',
            'price',
            'quantity_value',
            'quantity_unit',
        )

        return [
            {key: document[key] for key in fields if key in document}
            for document in documents
        ]

    return ToolRegistry([
        Tool(
            name="search_products",
            description="""
Search the product catalog using hybrid text and vector search. Use this tool when the user asks to find,
compare, or recommend products available in the catalog.

Search uses product names, descriptions, categories and brand names for text relevance, 
along with vector similarity based on product names and descriptions. Use query to describe the desired 
products and their characteristics, and top_k to control the maximum number of products returned.

Product descriptions contain structured sections such as:
- Short Description,
- What It Does & Benefits, 
- Instructions,
- Active Ingredients,
- Full List of Ingredients (INCI),
- Skin Type.
These sections provide information about product features, intended benefits, usage instructions, ingredients, 
and skin compatibility.

Examples:
- "Recommend products with vitamin C."
- "Find products from the Merlot brand."
- "Show products priced between 20 and 50."
- "What are the ingredients and benefits of this product?"

Only recommend products and prices supported by the search results. Never invent products or prices that 
are not present in the results.
""",
            fn=search_products,
            args_model=SearchProductsArgs,
        ),
    ])
