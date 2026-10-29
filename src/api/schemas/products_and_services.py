from typing import Literal

from pydantic import BaseModel


# Request
class Product(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    brand_name: str | None = None
    category_name: str | None = None
    subcategory_name: str | None = None
    price: float | None = None
    quantity_value: float | None = None
    quantity_unit: str | None = None


class Service(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    category_name: str | None = None
    subcategory_name: str | None = None
    price: float | None = None
    duration: int | None = None
    recovery_time: int | None = None


class Data(BaseModel):
    products: dict[str, list[Product]]
    services: dict[str, list[Service]]


class RequestProductsAndServices(BaseModel):
    data: Data


class ResponseProductsAndServices(BaseModel):
    status: Literal['ok']
    message: str
