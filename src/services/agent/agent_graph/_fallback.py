from typing import Any


def _has_tool_results(history: list[dict[str, Any]]) -> bool:
    """
    Return True if history contains at least one tool message.

    Synthetic error messages (skipped_due_to_termination_result,
    already_failed_result, repeated_action_result) also count: the model can
    still synthesize a final answer that explains the failure. Excluding them
    would silently skip finalization in error scenarios, which is exactly the
    case the feature is meant to handle. An empty content field is also
    counted; _deterministic_fallback independently decides whether a usable
    result is present.

    Args:
        history (list[dict[str, Any]]): Conversation history.

    Returns:
        bool: True if at least one tool message is present.
    """
    return any(msg.get('role') == 'tool' for msg in history)


def _deterministic_fallback(history: list[dict[str, Any]], last_tool_ok: str | None) -> str | None:
    """
    Build a deterministic answer from the most recent successful tool result,
    or from the last tool error message when no successful result exists.

    A tool result is considered successful based on the explicit ``last_tool_ok`` field maintained by ``execute_node``;
    this removes the fragile ``startswith('ERROR ')`` heuristic over message content.

    When no successful tool result is available, the last tool message in history (which, by elimination,
    must be an error: synthetic errors from skipped_due_to_termination_result / already_failed_result /
    repeated_action_result, or a real execution error) is used as a fallback explanation, so the caller returns
    a non-empty answer that describes the failure instead of a silent None.

    Returns None when no tool message is present at all, so the caller can leave answer=None and finalization=NONE.

    Args:
        history (list[dict[str, Any]]): Conversation history.
        last_tool_ok (str | None): Content of the most recent successful tool result, or None if no tool call has succeeded.

    Returns:
        str | None: A deterministic answer, or None if no tool message is present.
    """
    if last_tool_ok is not None:
        return (
            'The model could not produce a final answer within the step limit. Here is the most recent tool result:\n'
            f'{last_tool_ok}'
        )

    # No successful tool result. Use the last tool message as an explanation;
    # by elimination it must be an error (real or synthetic).
    for msg in reversed(history):
        if msg.get('role') != 'tool':
            continue

        content = msg.get('content') or ''

        return (
            'The model could not produce a final answer within the step limit. The last tool call failed:\n'
            f'{content}'
        )

    return None
