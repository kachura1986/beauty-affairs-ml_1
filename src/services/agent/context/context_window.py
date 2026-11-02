from dataclasses import dataclass
from typing import Any

# --- Limits ---------------------------------------------------------------
# NB: sizes are measured in *characters*, not tokens. This is a cheap approximation: the chars/tokens ratio varies
# by language and content, so treat ``max_estimated_chars`` as an upper-bound estimator, NOT a hard guarantee that
# the request fits the model's context. The ``RESERVED_OUTPUT_CHARS`` buffer absorbs the typical error.
MAX_MESSAGES = 40
MAX_ESTIMATED_CHARS = 60_000
RESERVED_OUTPUT_CHARS = 8_000  # room for the model's reply

# Only ``tool`` outputs are trimmed in place (head + tail). They are the main size driver, and their first 
# line (call_id / exit code) and last line (error / final value) are the useful parts. Cutting user/assistant text can 
# silently destroy meaning (reviewer note #5), so those are kept whole — if they don't fit, the whole turn is dropped.
PER_TOOL_MSG_LIMIT = 4_000

# Approximate char cost of a non-text multimodal part (image, audio, ...).
MULTIMODAL_PART_WEIGHT = 1_000


@dataclass(frozen=True)
class AgentContextWindow:
    """
    Ready context for a single LLM step.

    ``system`` and ``messages`` are separate provider channels: ``system`` is sent via ``chat(..., system_prompt=...)``
    / Responses API ``instructions``, ``messages`` — via ``input``.

    ``messages`` is not copied: the same object is passed to both ``chat`` and ``trace.step``, so ``window_hash``
    / ``window_msgs`` in the trace reflect exactly what was sent to the model.
    """

    system: str
    messages: list[dict[str, Any]]


class ContextOverflow(Exception):
    """
    System prompt + pinned turns or the current request alone exceed the budget.

    Raised only by ``build_context`` (and the private helpers it calls), called exclusively from ``decide_node``.
    ``decide_node`` MUST catch it and return ``TerminationReason.CONTEXT_OVERFLOW``; the run then terminates normally
    through ``finish_node`` and is reported to the caller via ``AgentResult.reason`` / ``ChatResponse.termination_reason``.
    """


# --- Helpers --------------------------------------------------------------
def _size(message: dict[str, Any]) -> int:
    """
    Approximate size of a message in characters.

    Multimodal (list) content sums text parts and assigns a fixed weight to every non-text part,
    so images cannot bypass the budget by hiding inside ``content``.
    
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

    Head is favoured (2/3) — call_id / exit code live there; tail (1/3) keeps the final error line or result value.
    Other roles are returned untouched — that is a deliberate trade-off, see module docstring.
    
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

    ``tool`` messages do not open turns — they belong to the preceding assistant turn and must stay grouped with
    the matching ``assistant(tool_calls)``, otherwise the provider API rejects the request.
    If the history does not start with ``user`` (e.g. it is already a tail of a longer session),
    the first group is simply the earliest available context and is treated as such.
    
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
        raise ContextOverflow('System prompt alone exceeds budget')

    return effective_system, history_budget


def _build_window(
        history: list[dict[str, Any]],
        history_budget_chars: int,
        max_messages: int = MAX_MESSAGES
) -> list[dict[str, Any]]:
    """
    Pack ``history`` into a list of ``input`` messages under ``history_budget_chars``.

    ``system_prompt`` is NOT part of the window — it goes to ``chat(system_prompt=...)``. ``history_budget_chars`` is 
    expected to ALREADY EXCLUDE the output reserve (see ``_prepare_context``); it is NOT re-subtracted here.

    Rules (in priority order):
      1. Client-supplied ``system`` messages are dropped (never part of ``input``).
      2. The earliest available turn is kept as the best available context (usually the task description,
         but the code does not assume it).
      3. The last turn is always kept — it is the current request; without it a reply would be meaningless.
         If it does not fit — overflow.
      4. Remaining turns are added newest-to-oldest until either ``max_messages`` or ``history_budget_chars``
         is hit. Iteration stops on the first turn that does not fit (``break``, not ``continue``):
         we keep a *continuous* recent tail. Skipping an oversized middle turn would create a gap in the
         reasoning chain, which is worse for multi-step tasks than losing older context.
      5. Oversized ``tool`` outputs are trimmed head + tail before grouping. User/assistant messages are
         kept whole.

    Pinning: only turn 0 and the last turn are pinned — this is a server-side decision. The client cannot
    influence pinning (a hostile client must not be able to pin everything and force a ``ContextOverflow``).

    Args:
        history (list[dict[str, Any]]): Conversation history to pack.
        history_budget_chars (int): Character budget for the packed history,
            already net of ``RESERVED_OUTPUT_CHARS`` and the system part.
        max_messages (int, optional): Maximum number of messages in the window.

    Raises:
        ContextOverflow: If pinned turns alone exceed the message or character budget.
    """
    # Step 1. sanitize history
    # Drop client system messages; trim only tool outputs (see _truncate_tool).
    messages = [_truncate_tool(m) for m in history if m.get("role") != "system"]
    if not messages:
        return []

    turns = _group_turns(messages)

    # Step 2. Pin the earliest and the last turn
    # Server-side pinning only. A client ``pinned`` flag is ignored.
    pinned_idx: set[int] = {0, len(turns) - 1}

    pinned_chars = sum(_size(m) for i in pinned_idx for m in turns[i])
    pinned_msgs = sum(len(turns[i]) for i in pinned_idx)

    if pinned_msgs > max_messages:
        raise ContextOverflow("Pinned turns exceed message limit")

    if pinned_chars > history_budget_chars:
        # Includes the "current request alone is too large" case — this is a deliberate, documented contract.
        raise ContextOverflow("Mandatory turns exceed budget")

    budget = history_budget_chars - pinned_chars
    keep: set[int] = set(pinned_idx)
    kept_msgs = pinned_msgs
    used = 0

    # Step 3. Extend with a continuous recent tail
    # Walk backwards so newest turns win. ``break`` is intentional — see docstring step 4. A hostile/oversized
    # middle turn is not replaced by an older small one: continuity of the recent tail matters more.
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

    # Step 4: restore chronological order
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

    ``system`` and ``messages`` in the result go to different provider channels — see :class:`AgentContextWindow`.

    ``max_messages`` exists for testability; in production the default``MAX_MESSAGES`` is always used.

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
