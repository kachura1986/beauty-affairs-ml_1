from dataclasses import dataclass, field

CANONICAL_ROLES = {'system', 'user', 'assistant', 'tool'}


@dataclass(slots=True)
class Message:
    """
    Represent a single chat message.

    Attributes:
        role (str): Logical role of the message author.
        content (str): Text content of the message.
        tool_calls (list[dict[str, object]] | None): List of tool calls.
        tool_call_id (str | None): ID of the tool call.
        name (str | None): Name of the message author.
    """

    role: str
    content: str
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class UnknownRoleError(ValueError):
    """Raise an error when a message contains an unsupported role."""

    pass


@dataclass
class Reply:
    """
    Represent a normalized model response.

    Attributes:
        text (str): Text content of the response.
        tool_calls (list[dict[str, object]]): Tool calls requested by the model.
        model (str): Name of the model that generated the response.
        in_tokens (int): Number of input tokens.
        out_tokens (int): Number of output tokens.
        cost_usd (float): Estimated response cost in US dollars.
    """

    text: str = ''
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    model: str = 'fake'
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
