from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    postcondition: Callable[[str], tuple[bool, str]] | None = None

    max_steps: int = Field(default=16, gt=0)
    max_seconds: float = Field(default=300.0, gt=0)
    max_tokens: int = Field(default=200_000, gt=0)
    max_cost_usd: float = Field(default=0.50, gt=0)
