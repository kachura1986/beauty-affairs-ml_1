from typing import Any

from services.agent.tools import ToolResult
from services.llm import ToolCall


def build_observation_message(text: str) -> dict[str, str]:
    """
    Build a harness observation message for the model.

    CONTRACT: uses ``role='user'`` because it is the only role with no structural requirements
    (no ``tool_call_id``, no fixed position), is not filtered out by
    :func:`~services.agent.context.build_context` (unlike ``system``), and does not falsely
    attribute the text to the model (unlike ``assistant``). The ``[observation]`` prefix is a
    semantic marker — the chat protocol has no dedicated "harness" role.

    Args:
        text (str): Observation text to deliver to the model.

    Returns:
        dict[str, str]: A ``role='user'`` message prefixed with ``[observation]``.
    """
    return {'role': 'user', 'content': f'[observation] {text}'}


def build_tool_message(call: ToolCall, res: ToolResult) -> dict[str, Any]:
    """
    Build a tool result message for the model.

    CONTRACT: ``tool_call_id`` falls back to the tool name when the call has no explicit ``id``,
    so the message stays paired with the originating assistant tool_call.

    Args:
        call (ToolCall): Tool call dictionary containing at least ``name`` and optionally ``id``.
        res (ToolResult): Result of executing the tool call.

    Returns:
        dict[str, Any]: A ``role='tool'`` message with the tool name, resolved ``tool_call_id``, and rendered result.
    """
    return {
        'role': 'tool',
        'name': call['name'],
        'tool_call_id': call.get('id', call['name']),
        'content': res.to_model()
    }
