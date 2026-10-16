import logging

from fastapi import APIRouter, Depends

from api.schemas.health_check import HealthResponse
from core.config import settings
from dependencies import get_logger

router = APIRouter()


@router.get('', response_model=HealthResponse)
async def health_check(logger: logging.Logger = Depends(get_logger)) -> HealthResponse:
    """Check that the API is running and ready to receive requests.

    Args:
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        HealthResponse: Status of the application.
    """
    logger.info(f"'{settings.app_title}' is running")

    return HealthResponse(status="ok", app=settings.app_title)
