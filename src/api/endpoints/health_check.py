import logging

from fastapi import APIRouter

from api.schemas.health_check import HealthResponse
from core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check that the API is running and ready to receive requests.

    Returns:
        HealthResponse: Status of the application.
    """
    logger.info(f"'{settings.APP_TITLE}' is running")

    return HealthResponse(status="ok", app=settings.APP_TITLE)
