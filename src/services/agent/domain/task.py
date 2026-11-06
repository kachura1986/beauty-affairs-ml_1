from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .budget import BudgetLimits


class Task(BaseModel):
    """
    A single agent_graph task.

    CONTRACT: ``initial_messages`` is the append-only starting history — the run may add to it, never rewrite it.
    ``postcondition``, when set, is called on the final answer and must return ``(ok, note)``.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(description="System prompt used to build the context window.")
    initial_messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Append-only conversation history the run starts from.",
    )

    postcondition: Callable[[str], tuple[bool, str]] | None = Field(
        default=None,
        description="Optional final-answer verifier; returns (ok, note).",
    )
    budget_limits: BudgetLimits = Field(..., description="Resource limits used to build the run budget.")
