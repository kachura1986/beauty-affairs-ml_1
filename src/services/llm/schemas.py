from dataclasses import dataclass, field
from typing import Any, TypedDict

CANONICAL_ROLES = {'system', 'user', 'assistant', 'tool'}


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]


class UnknownRoleError(ValueError):
    """Raise an error when a message contains an unsupported role."""

    pass


@dataclass
class Reply:
    """
    Represent a normalized model response.

    Attributes:
        text (str): Text content of the response.
        tool_calls (list[ToolCall]): Tool calls requested by the model.
        model (str): Name of the model that generated the response.
        in_tokens (int): Number of input tokens.
        out_tokens (int): Number of output tokens.
        cost (float): Estimated response cost, in the currency of the configured tariff (typically USD).
    """

    text: str = ''
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = 'fake'
    in_tokens: int = 0
    out_tokens: int = 0
    cost: float = 0.0
