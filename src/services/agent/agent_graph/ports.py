from typing import Any, Protocol

from services.llm import Reply

from ..context import AgentContextWindow
from ..domain import Budget, Finalization, TerminationReason


class StepRunner(Protocol):
    """One bounded LLM decision step."""

    async def run(
            self, *, budget: Budget, context_window: AgentContextWindow,
    ) -> tuple[Reply | None, TerminationReason | None]: ...


class Finalizer(Protocol):
    """One bounded finalization call after max_steps is exhausted."""

    async def run(
            self, *, budget: Budget, history: list[dict[str, Any]], last_tool_ok: str | None,
    ) -> tuple[str | None, Finalization]: ...
