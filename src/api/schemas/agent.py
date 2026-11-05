from typing import Literal

from pydantic import BaseModel, Field

from services.agent.domain import Finalization, TerminationReason


class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., description="Dialog history with roles 'user' and 'assistant'", )


class ChatUsage(BaseModel):
    steps: int
    tokens: int
    cost_usd: float
    seconds: float


class ChatResponse(BaseModel):
    run_id: str = Field(description="Run identifier; matches the trace file stem.")
    role: Literal['assistant']
    content: str | None
    termination_reason: TerminationReason
    finalization: Finalization = Field(
        description=(
            "How the answer was produced. LLM — synthesized by the model in finalize_node;  FALLBACK — deterministic "
            "fallback from the last tool result; NONE — no finalization was attempted."
        ),
    )
    usage: ChatUsage
