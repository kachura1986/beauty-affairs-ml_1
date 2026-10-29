from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Dialog history with roles 'user' and 'assistant'",
    )
