from typing import Any

from services.agent.tools import ToolResult


def build_observation_message(text: str) -> dict[str, str]:
    """
    Build a harness observation message for the model.

    Uses ``role='user'`` because it is the only role with no structural requirements
    (no ``tool_call_id``, no fixed position), is not filtered out by :meth:`Task.build_window` (unlike ``system``),
    does not falsely attribute the text to the model (unlike ``assistant``), and is treated as
    guidance rather than data (unlike ``tool``, see SYSTEM_PROMPT rule 4). The ``[observation]`` prefix is
    a semantic marker, since the chat protocol has no dedicated "harness" role.

    Args:
        text (str): Observation text to deliver to the model.

    Returns:
        dict[str, str]: A ``role='user'`` message prefixed with ``[observation]``.
    """
    return {'role': 'user', 'content': f'[observation] {text}'}


def build_tool_message(call: dict, res: ToolResult) -> dict[str, Any]:
    """
    Build a tool result message for the model.

    Links the result to the originating tool call via ``tool_call_id``, which falls back to the tool name
    when the call has no explicit ``id``. The content is rendered by :meth:`ToolResult.to_model`.

    Args:
        call (dict): Tool call dictionary containing at least ``name`` and optionally ``id``.
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
