from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        description="Dialog history with roles 'system', 'user', 'assistant' and 'tool'",
    )
