from pydantic import BaseModel, Field

from services.llm import Message


class ChatRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="Model identifier, e.g. 'gemini:gemini-2.5-flash' or 'ollama:llama3'",
        examples=['ollama:qwen3:8b'],
    )
    messages: list[Message] = Field(
        ...,
        description="Dialog history with roles 'system', 'user', 'assistant' and 'tool'",
    )
