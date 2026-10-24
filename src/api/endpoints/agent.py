import json
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas.agent import ChatRequest
from application import AgentService
from dependencies import get_agent_service, get_logger
from services.llm import UnknownRoleError

router = APIRouter()


@router.post('/chat/stream')
async def agent_chat_stream(
        request: ChatRequest,
        agent_service: AgentService = Depends(get_agent_service),
        logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    """
    Streams the agent response using server-sent events.

    Args:
        request (ChatRequest): Chat request containing messages and model identifier.
        agent_service (AgentService): Orchestrator responsible for executing agent requests.
        logger (logging.Logger): Logger used for reporting request errors.

    Returns:
        StreamingResponse: Streaming HTTP response containing server-sent events.
    """
    request_id = str(uuid4())
    started = time.perf_counter()

    def sse_event(event_type: str, **payload: object) -> str:
        """
        Builds a server-sent event payload.

        Args:
            event_type (str): Type of the server-sent event.
            payload (object): Additional data included in the event.

        Returns:
            str: Serialized server-sent event.
        """
        event = {'type': event_type, **payload}
        serialized_event = f'data: {json.dumps(event, ensure_ascii=False)}\n\n'

        return serialized_event

    async def stream_generator() -> object:
        """
        Generates server-sent events for the agent response.

        Yields:
            str: Serialized server-sent event.
        """
        try:
            result = await agent_service.run(request.messages)

            if result.answer:
                yield sse_event('token', content=result.answer)

            yield sse_event(
                'done',
                request_id=request_id,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        except UnknownRoleError as exc:
            logger.warning(f"Invalid role, request_id={request_id}: {exc}")
            yield sse_event('error', request_id=request_id, message=str(exc))

        except ValueError as exc:
            logger.warning(f"Invalid agent request, request_id={request_id}: {exc}")
            yield sse_event('error', request_id=request_id, message=str(exc))

        except Exception:
            logger.exception(f"Agent streaming failed, request_id={request_id}")
            yield sse_event(
                'error',
                request_id=request_id,
                message="Ошибка генерации. Повторите запрос позже.",
            )

    streaming_response = StreamingResponse(
        stream_generator(),
        media_type='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Request-ID': request_id,
        },
    )

    return streaming_response
