from fastapi import APIRouter, Depends

from api.schemas.agent import ChatRequest, ChatResponse, ChatUsage
from dependencies import get_agent_service
from use_cases import AgentService

router = APIRouter()


@router.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)) -> ChatResponse:
    """
    Executes the agent request and returns the response.

    Args:
        request (ChatRequest): Chat request containing messages.
        agent_service (AgentService): Orchestrator responsible for executing agent requests.

    Returns:
        ChatResponse: Structured agent response with run_id, role, content, termination reason,
        finalization and usage stats.
    """
    result = await agent_service.run(request.messages)

    return ChatResponse(
        run_id=result.run_id,
        role='assistant',
        content=result.answer,
        termination_reason=result.reason,
        finalization=result.finalization,
        usage=ChatUsage(
            steps=result.steps,
            tokens=result.tokens,
            cost=result.cost,
            seconds=result.seconds,
        ),
    )
