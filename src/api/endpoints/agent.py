import logging

from fastapi import APIRouter, Depends

from api.schemas.agent import ChatRequest
from dependencies import get_agent_service, get_logger
from use_cases import AgentService

router = APIRouter()


@router.post('/chat')
async def chat(
        request: ChatRequest,
        agent_service: AgentService = Depends(get_agent_service),
        logger: logging.Logger = Depends(get_logger),
) -> dict:
    """
    Executes the agent request and returns the response.

    Args:
        request (ChatRequest): Chat request containing messages and model identifier.
        agent_service (AgentService): Orchestrator responsible for executing agent requests.
        logger (logging.Logger): Logger used for reporting request errors.

    Returns:
        dict: Agent response containing the answer, request ID, and latency.
    """
    result = await agent_service.run(request.messages)
    logger.info(request.messages[-1])

    return {'answer': result.answer}