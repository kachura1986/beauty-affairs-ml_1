import logging

from fastapi import APIRouter, Depends

from core.retry_strategy import retry_strategy
from dependencies import get_logger
from ..schemas.products_and_services import RequestProductsAndServicesSchema

router = APIRouter()


@retry_strategy
@router.post('/update', response_model=str)
async def update(
        data: RequestProductsAndServicesSchema,
        logger: logging.Logger = Depends(get_logger),
) -> str:
    """
    Update products and services in the search index.

    Args:
        data (RequestProductsAndServicesSchema):
            - **products** (dict): Dictionary of products to update.
            - **services** (dict): Dictionary of services to update.
        logger (logging.Logger): Logger used for reporting request errors.

    Returns:
        str: A response message.

    Raises:
        HTTPException: If an unexpected error occurs during processing, a 500 error is raised.
    """
    products = {f'products_{key}': value for key, value in data.model_dump()['data']['products'].items()}
    services = {f'services_{key}': value for key, value in data.model_dump()['data']['services'].items()}

    logger.info('Products: %s', products)
    logger.info('Services: %s', services)

    return 'Products and services have been updated in the search index'


@retry_strategy
@router.post('/delete', response_model=str)
async def delete(
        data: RequestProductsAndServicesSchema,
        logger: logging.Logger = Depends(get_logger),
) -> str:
    """
    Delete products and services in search index.

    Args:
        data (RequestProductsAndServicesSchema):
            - **products** (dict): Dictionary of products to delete.
            - **services** (dict): Dictionary of services to delete.
        logger (logging.Logger): Logger used for reporting request errors.

    Returns:
        str: A response message.

    Raises:
        HTTPException: If an unexpected error occurs during the scaling process, a 500 error is raised.
    """
    products = {f'products_{key}': value for key, value in data.model_dump()['data']['products'].items()}
    services = {f'services_{key}': value for key, value in data.model_dump()['data']['services'].items()}

    logger.info('Products: %s', products)
    logger.info('Services: %s', services)

    return "Products and services have been deleted in the search index"
