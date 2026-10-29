import logging

from fastapi import APIRouter, Depends

from core.retry_strategy import retry_strategy
from dependencies import get_index_service
from use_cases.index_service import IndexService
from ..schemas.products_and_services import RequestProductsAndServices, ResponseProductsAndServices

router = APIRouter()
logger = logging.getLogger(__name__)


@retry_strategy(exceptions=(ConnectionError, TimeoutError))
async def _upsert_documents_with_retry(index_service: IndexService, products: dict, services: dict) -> None:
    await index_service.upsert_documents(products, services)


@retry_strategy(exceptions=(ConnectionError, TimeoutError))
async def _delete_documents_with_retry(index_service: IndexService, products: dict, services: dict) -> None:
    await index_service.delete_documents(products, services)


def log_items(items: dict, item_type: str) -> None:
    items_info = [
        {'id': item.get('id'), 'name': item.get('name')}
        for item_list in items.values()
        for item in item_list
    ]

    logger.info(
        f"{item_type} count: {sum(len(value) for value in items.values())}, "
        f"{item_type.lower()}: {items_info}",
    )


def _prefix_and_log(data: RequestProductsAndServices) -> tuple[dict, dict]:
    payload = data.model_dump()['data']

    products = {f'products_{key}': value for key, value in payload['products'].items()}
    services = {f'services_{key}': value for key, value in payload['services'].items()}

    log_items(products, "Products")
    log_items(services, "Services")

    return products, services


@router.post('/update', response_model=ResponseProductsAndServices)
async def update(
        data: RequestProductsAndServices,
        index_service: IndexService = Depends(get_index_service)
) -> ResponseProductsAndServices:
    """
    Update products and services in the search index.

    Args:
        data (RequestProductsAndServices):
            - **data.products** (dict): Dictionary of products to update.
            - **data.services** (dict): Dictionary of services to update.
        index_service (IndexService): Index service used for updating products and services.

    Returns:
        ResponseProductsAndServices: A response message.
    """
    products, services = _prefix_and_log(data)

    await _upsert_documents_with_retry(index_service, products, services)

    return ResponseProductsAndServices(
        status="ok",
        message="Products and services have been updated in the search index"
    )


@router.post('/delete', response_model=ResponseProductsAndServices)
async def delete(
        data: RequestProductsAndServices,
        index_service: IndexService = Depends(get_index_service)
) -> ResponseProductsAndServices:
    """
    Delete products and services in search index.

    Args:
        data (RequestProductsAndServices):
            - **data.products** (dict): Dictionary of products to delete.
            - **data.services** (dict): Dictionary of services to delete.
        index_service (IndexService): Index service used for updating products and services.

    Returns:
        ResponseProductsAndServices: A response message.
    """
    products, services = _prefix_and_log(data)

    await _delete_documents_with_retry(index_service, products, services)

    return ResponseProductsAndServices(
        status="ok",
        message="Products and services have been deleted in the search index"
    )
