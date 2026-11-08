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
            system_prompt: str | None = None,
    ) -> Reply:
        ...

    def is_transient_error(self, exc: Exception) -> bool:
        """Return True iff a retry of ``exc`` has a non-trivial chance of succeeding.

        CONTRACT: the implementation owns provider-specific classification (SDK wrapping, transport cause chain,
        status codes). The agent layer must not depend on those internals — it delegates here through
        ``_classify_llm_exception``.

        Args:
            exc (Exception): Exception raised by ``chat`` or its wrapper.

        Returns:
            bool: True if a retry has a non-trivial chance of succeeding.
        """
        ...
