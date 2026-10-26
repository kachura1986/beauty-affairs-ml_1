from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str
    initial_messages: list[dict[str, Any]] = Field(default_factory=list)
    postcondition: Callable[[str], tuple[bool, str]] | None = None

    max_steps: int = Field(default=16, gt=0)
    max_seconds: float = Field(default=300.0, gt=0)
    max_tokens: int = Field(default=200_000, gt=0)
    max_cost_usd: float = Field(default=0.50, gt=0)

    def build_window(self, history: list[dict[str, Any]], warning: str | None = None) -> list[dict[str, Any]]:
        window: list[dict[str, Any]] = [
            {
                'role': 'system',
                'content': self.system_prompt,
            }
        ]

        if warning:
            window.append({
                'role': 'system',
                'content': warning,
            })

        # Do not add system prompts from the client history
        window.extend(message for message in history if message.get('role') != 'system')

        return window
