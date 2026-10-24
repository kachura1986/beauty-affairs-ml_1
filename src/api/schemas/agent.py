from pydantic import BaseModel, Field

from services.llm import Message


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Dialog history with roles 'system', 'user', 'assistant' and 'tool'",
    )
