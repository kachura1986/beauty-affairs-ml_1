from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .schemas import Reply


class LLMModel(Protocol):
    """Define the application-level interface for asynchronous LLM models."""

    @property
    def model_name(self) -> str:
        ...

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
    ) -> Reply:
        ...
