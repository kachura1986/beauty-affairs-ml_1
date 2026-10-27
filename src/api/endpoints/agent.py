from fastapi import APIRouter, Depends

from api.schemas.agent import ChatRequest
from dependencies import get_agent_service
from use_cases import AgentService

router = APIRouter()


@router.post('/chat')
async def chat(request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)) -> dict:
    """
    Executes the agent request and returns the response.

    Args:
        request (ChatRequest): Chat request containing messages and model identifier.
        agent_service (AgentService): Orchestrator responsible for executing agent requests.

    Returns:
        dict: Agent response containing the answer, request ID, and latency.
    """
    result = await agent_service.run(request.messages)

    return {'answer': result.answer}