import re
from dataclasses import dataclass
from typing import Any

# --- Limits ---------------------------------------------------------------
# Sizes are in *characters*, not tokens: an upper-bound estimator, not a hard guarantee that the
# request fits the model's context. RESERVED_OUTPUT_CHARS absorbs the typical error.
MAX_MESSAGES = 40
MAX_ESTIMATED_CHARS = 60_000
RESERVED_OUTPUT_CHARS = 8_000

# Only tool outputs are trimmed in place (head 2/3 + tail 1/3): they are the main size driver, and their
# first lines (call_id / exit code) and last line (error / final value) carry the useful info. User/assistant
# text is never split — cutting it silently destroys meaning; whole turns are dropped instead.
PER_TOOL_MSG_LIMIT = 4_000

# Approximate char cost of a non-text multimodal part (image, audio, ...).
MULTIMODAL_PART_WEIGHT = 1_000

# Machine-readable codes of ContextOverflow are snake_case, lowercase, ASCII.
_CODE_RE = re.compile(r'[a-z0-9_]+')


@dataclass(frozen=True)
class AgentContextWindow:
    """
    Ready context for a single LLM step; ``system`` and ``messages`` go to separate provider channels.

    CONTRACT: ``messages`` is not copied — the same object is passed to both ``chat`` and ``trace.step``,
    so the trace's ``window_hash`` / ``window_msgs`` reflect exactly what was sent to the model.
    """

    system: str
    messages: list[dict[str, Any]]


class ContextOverflow(Exception):
    """
    System prompt + pinned turns or the current request alone exceed the budget.

    CONTRACT: raised only by ``build_context`` (called exclusively from ``decide_node``). ``decide_node``
    MUST catch it and terminate with ``TerminationReason.CONTEXT_OVERFLOW``; the run then finishes normally
    through ``finish_node``. The first positional argument is the **code** (must match ``[a-z0-9_]+``) —
    that is what ``Trace.error(message=...)`` records; the second is human-readable text for logs only.

    Attributes:
        code: Machine-readable, PII-free identifier of the overflow subtype.
        message: Human-readable detail (``str(exc)``). Logs only, never trace.
    """

    def __init__(self, code: str, message: str) -> None:
        if not _CODE_RE.fullmatch(code):
            raise ValueError(
                f"ContextOverflow code must match [a-z0-9_]+, got {code!r}. "
                f"Human-readable text belongs in the second positional argument."
            )

        super().__init__(message)
        self.code = code


# --- Helpers --------------------------------------------------------------
def _size(message: dict[str, Any]) -> int:
    """
    Approximate size of a message in characters.

    CONTRACT: multimodal (list) content sums text parts and assigns ``MULTIMODAL_PART_WEIGHT`` to every
    non-text part — otherwise images could bypass the budget by hiding inside ``content``.

    Args:
        message (dict[str, Any]): The message to measure.

    Returns:
        The size of the message in characters.
    """
    content = message.get('content')
    if isinstance(content, str):
        return len(content)

    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue

            if part.get('type') == 'text':
                total += len(part.get('text') or '')
            else:
                total += MULTIMODAL_PART_WEIGHT

        return total

    return 0


def _truncate_tool(message: dict[str, Any]) -> dict[str, Any]:
    """
    Trim a ``tool`` message to ``PER_TOOL_MSG_LIMIT``, keeping head + tail.

    CONTRACT: only ``role='tool'`` with string content is touched; every other role is returned as-is
    (splitting user/assistant text would silently corrupt meaning — see module comment).

    Args:
        message (dict[str, Any]): The message to truncate.

    Returns:
        The truncated message.
    """
    if message.get('role') != 'tool':
        return message
    if _size(message) <= PER_TOOL_MSG_LIMIT:
        return message

    content = message.get('content')
    if not isinstance(content, str):
        return message

    head = PER_TOOL_MSG_LIMIT * 2 // 3
    tail = PER_TOOL_MSG_LIMIT // 3
    trimmed = (
        f'{content[:head]}\n'
        f'…[truncated {len(content) - head - tail} chars]…\n'
        f'{content[-tail:]}'
    )

    return {**message, 'content': trimmed}


def _group_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Group messages into turns; every ``user`` message opens a new turn.

    CONTRACT: ``tool`` messages never open a turn — they belong to the preceding assistant turn and must
    stay grouped with the matching ``assistant(tool_calls)``, otherwise the provider API rejects the request.
    A history not starting with ``user`` groups the leading block as its own first turn.

    Args:
        messages (list[dict[str, Any]]): The messages to group.

    Returns:
        list[list[dict[str, Any]]]: The grouped messages.
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get('role') == 'user' and current:
            turns.append(current)
            current = []
        current.append(msg)

    if current:
        turns.append(current)

    return turns


def _prepare_context(system_prompt: str, warning: str | None) -> tuple[str, int]:
    """
    Combine the task system prompt with an optional soft-budget warning
    and compute the character budget available for history messages.

    The returned ``history_budget_chars`` is already net of ``RESERVED_OUTPUT_CHARS`` and the system part.

    Args:
        system_prompt (str): Task-level system prompt.
        warning (str | None): Optional soft-budget warning to append.

    Returns:
        tuple[str, int]: ``(effective_system, history_budget_chars)``.

    Raises:
        ContextOverflow: If the system part alone does not fit the context.
    """
    effective_system = system_prompt.strip()
    if warning:
        effective_system = f'{effective_system}\n\n{warning.strip()}'

    history_budget = MAX_ESTIMATED_CHARS - RESERVED_OUTPUT_CHARS - len(effective_system)
    if history_budget <= 0:
        raise ContextOverflow('system_prompt_too_large', "System prompt alone exceeds budget")

    return effective_system, history_budget


def _build_window(
        history: list[dict[str, Any]],
        history_budget_chars: int,
        max_messages: int = MAX_MESSAGES
) -> list[dict[str, Any]]:
    """
    Pack ``history`` into a list of ``input`` messages under ``history_budget_chars``.

    ``system_prompt`` is NOT part of the window — it goes to ``chat(system_prompt=...)``.
    CONTRACT: ``history_budget_chars`` is expected to ALREADY EXCLUDE the output reserve (see
    ``_prepare_context``); it is NOT re-subtracted here. Pinning is server-side only — a client
    ``pinned`` flag is ignored so a hostile client cannot force an overflow by pinning everything.
    Remaining turns are added newest-to-oldest with ``break`` (not ``continue``): a *continuous*
    recent tail is kept; skipping an oversized middle turn would open a gap in the reasoning chain.

    Args:
        history (list[dict[str, Any]]): Conversation history to pack.
        history_budget_chars (int): Character budget for the packed history, already net of
            ``RESERVED_OUTPUT_CHARS`` and the system part.
        max_messages (int, optional): Maximum number of messages in the window.

    Raises:
        ContextOverflow: If pinned turns alone exceed the message or character budget
            (includes the "current request alone is too large" case).
    """
    # Drop client system messages; trim only tool outputs (see _truncate_tool).
    messages = [_truncate_tool(m) for m in history if m.get("role") != "system"]
    if not messages:
        return []

    turns = _group_turns(messages)

    # Pin earliest + last turn (server-side only; client ``pinned`` flag is ignored).
    pinned_idx: set[int] = {0, len(turns) - 1}

    pinned_chars = sum(_size(m) for i in pinned_idx for m in turns[i])
    pinned_msgs = sum(len(turns[i]) for i in pinned_idx)

    if pinned_msgs > max_messages:
        raise ContextOverflow('pinned_turns_too_many', "Pinned turns exceed message limit")

    if pinned_chars > history_budget_chars:
        raise ContextOverflow('pinned_turns_too_large', "Mandatory turns exceed budget")

    budget = history_budget_chars - pinned_chars
    keep: set[int] = set(pinned_idx)
    kept_msgs = pinned_msgs
    used = 0

    # Extend with a continuous recent tail. ``break`` is intentional — see docstring.
    for i in range(len(turns) - 2, 0, -1):
        if i in keep:
            continue
        turn = turns[i]
        turn_size = sum(_size(m) for m in turn)
        turn_len = len(turn)
        if kept_msgs + turn_len > max_messages:
            break
        if used + turn_size > budget:
            break
        keep.add(i)
        used += turn_size
        kept_msgs += turn_len

    window: list[dict[str, Any]] = []
    for i in sorted(keep):
        window.extend(turns[i])

    return window


def build_context(
        system_prompt: str,
        warning: str | None,
        history: list[dict[str, Any]],
        max_messages: int = MAX_MESSAGES,
) -> AgentContextWindow:
    """
    Assemble the context for a single LLM step: effective system prompt plus the packed history.

    ``system`` and ``messages`` in the result go to different provider channels — see
    :class:`AgentContextWindow`. ``max_messages`` exists for testability; production always uses
    the ``MAX_MESSAGES`` default.

    Args:
        system_prompt (str): Task-level system prompt.
        warning (str | None): Optional soft-budget warning to append.
        history (list[dict[str, Any]]): Conversation history to pack.
        max_messages (int, optional): Maximum number of messages in the window.

    Returns:
        AgentContextWindow: System prompt and packed history, kept separately.

    Raises:
        ContextOverflow: If the system part or the pinned turns do not fit the budget.
    """
    effective_system, history_budget = _prepare_context(system_prompt, warning)
    messages = _build_window(history=history, history_budget_chars=history_budget, max_messages=max_messages)

    return AgentContextWindow(system=effective_system, messages=messages)
