from typing import Literal

from pydantic import BaseModel, Field

from services.agent.domain import TerminationReason


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
    role: Literal['assistant']
    content: str | None
    termination_reason: TerminationReason
    usage: ChatUsage
