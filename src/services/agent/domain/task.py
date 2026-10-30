from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .budget import BudgetLimits


def _default_budget_limits() -> BudgetLimits:
    return BudgetLimits(
        max_steps=16,
        max_seconds=300.0,
        max_tokens=200_000,
        max_cost_usd=0.50,
    )


class Task(BaseModel):
    """
    A single agent_graph task.

    Attributes:
        system_prompt (str): System prompt used to build the context window.
        initial_messages (list[dict[str, Any]]): Append-only conversation history the run starts from.
        postcondition (Callable[[str], tuple[bool, str]] | None): Optional final-answer verifier; returns ``(ok, note)``.
        budget_limits (BudgetLimits): Resource limits used to build the run budget.
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
    budget_limits: BudgetLimits = Field(
        default_factory=_default_budget_limits,
        description="Resource limits used to build the run budget.",
    )
