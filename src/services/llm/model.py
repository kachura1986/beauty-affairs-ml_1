from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from .schemas import Message, Reply


class LLMModel(Protocol):
    """
    Define the application-level interface for asynchronous LLM models.
    """

    @property
    def model_name(self) -> str:
        ...

    async def generate_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> str:
        ...

    async def get_streaming_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> AsyncIterator[str]:
        ...

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
            stop: Sequence[str] | None = None,
    ) -> Reply:
        ...
