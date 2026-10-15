import logging

from fastapi import APIRouter, Depends

from api.schemas.health_check import HealthResponse, ProviderHealth
from application import LLMModelResolver
from core.config import settings
from dependencies import get_logger, get_model_resolver

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


@router.get('/providers')
async def provider_health(llm_resolver: LLMModelResolver = Depends(get_model_resolver)) -> ProviderHealth:
    """
    Returns connectivity status and latency for configured LLM providers.

    Args:
        llm_resolver(LLMModelResolver): Orchestrator instance for LLM operations.

    Returns:
        dict[str, Any]: Dictionary with status and provider information.
    """
    providers_status = await llm_resolver.get_provider_health()

    return ProviderHealth(status=providers_status['status'], providers=providers_status['providers'])
