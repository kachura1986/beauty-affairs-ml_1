from typing import Any


def _has_tool_results(history: list[dict[str, Any]]) -> bool:
    """
    Return True if history contains at least one tool message.

    CONTRACT: synthetic errors (skipped, already_failed, repeated_action) and empty-content
    tool messages also count — finalization must run when every call failed so the model can
    explain it. Whether a usable result exists is decided separately by ``_deterministic_fallback``.

    Args:
        history (list[dict[str, Any]]): Conversation history.

    Returns:
        bool: True if at least one tool message is present.
    """
    return any(msg.get('role') == 'tool' for msg in history)


def _deterministic_fallback(history: list[dict[str, Any]], last_tool_ok: str | None) -> str | None:
    """
    Build a deterministic answer from the most recent successful tool result,
    or from the last tool error when no successful result exists.

    CONTRACT: success is determined by the explicit ``last_tool_ok`` field (maintained by
    ``execute_node``), not by a heuristic over message content. Returns None when no tool
    message is present at all.

    Args:
        history (list[dict[str, Any]]): Conversation history.
        last_tool_ok (str | None): Content of the most recent successful tool result, or None.

    Returns:
        str | None: A deterministic answer, or None if no tool message is present.
    """
    if last_tool_ok is not None:
        return (
            'The model could not produce a final answer within the step limit. Here is the most recent tool result:\n'
            f'{last_tool_ok}'
        )

    # No successful tool result: by elimination the last tool message must be an error (real or synthetic).
    for msg in reversed(history):
        if msg.get('role') != 'tool':
            continue

        content = msg.get('content') or ''

        return (
            'The model could not produce a final answer within the step limit. The last tool call failed:\n'
            f'{content}'
        )

    return None