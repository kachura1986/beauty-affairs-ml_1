from typing import Any

# --- Limits ---------------------------------------------------------------
# NB: sizes are measured in *characters*, not tokens. This is a cheap approximation: the chars/tokens ratio varies
# by language and content, so treat ``max_estimated_chars`` as an upper-bound estimator, NOT a hard guarantee that
# the request fits the model's context. The ``RESERVED_OUTPUT_CHARS`` buffer absorbs the typical error.
MAX_MESSAGES = 40
MAX_ESTIMATED_CHARS = 60_000
RESERVED_OUTPUT_CHARS = 8_000  # room for the model's reply

# Only ``tool`` outputs are trimmed in place (head + tail). They are the main size driver, and
# their first line (call_id / exit code) and last line (error / final value) are the useful parts.
# Cutting user/assistant text can silently destroy meaning (reviewer note #5), so those are kept
# whole — if they don't fit, the whole turn is dropped.
PER_TOOL_MSG_LIMIT = 4_000

# Approximate char cost of a non-text multimodal part (image, audio, ...).
MULTIMODAL_PART_WEIGHT = 1_000


class ContextOverflow(Exception):
    """
    Pinned content or the current request alone exceeds the budget.

    Contract for callers:
      * MUST catch this and translate into a user-facing message
        (e.g. "please start a new dialog");
      * MUST NOT let it bubble up as a 500 server error.
    """


# --- Helpers --------------------------------------------------------------
def _size(msg: dict[str, Any]) -> int:
    """
    Approximate size of a message in characters.

    Multimodal (list) content sums text parts and assigns a fixed weight to every non-text part,
    so images cannot bypass the budget by hiding inside ``content``.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return len(content)

    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue

            if part.get("type") == "text":
                total += len(part.get("text") or "")
            else:
                total += MULTIMODAL_PART_WEIGHT

        return total

    return 0


def _truncate_tool(msg: dict[str, Any]) -> dict[str, Any]:
    """
    Trim a ``tool`` message to ``PER_TOOL_MSG_LIMIT``, keeping head + tail.

    Head is favoured (2/3) — call_id / exit code live there; tail (1/3) keeps the final error line or result value.
    Other roles are returned untouched — that is a deliberate trade-off, see module docstring.
    """
    if msg.get("role") != "tool":
        return msg
    if _size(msg) <= PER_TOOL_MSG_LIMIT:
        return msg

    content = msg.get("content")
    if not isinstance(content, str):
        return msg

    head = PER_TOOL_MSG_LIMIT * 2 // 3
    tail = PER_TOOL_MSG_LIMIT // 3
    trimmed = (
        f"{content[:head]}\n"
        f"…[truncated {len(content) - head - tail} chars]…\n"
        f"{content[-tail:]}"
    )

    return {**msg, "content": trimmed}


def _group_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Group messages into turns; every ``user`` message opens a new turn.

    ``tool`` messages do not open turns — they belong to the preceding assistant turn and must stay grouped with
    the matching ``assistant(tool_calls)``, otherwise the provider API rejects the request.
    If the history does not start with ``user`` (e.g. it is already a tail of a longer session),
    the first group is simply the earliest available context and is treated as such.
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)

    if current:
        turns.append(current)

    return turns


# Build window
def build_window(
        system_prompt: str,
        history: list[dict[str, Any]],
        warning: str | None = None,
        max_messages: int = MAX_MESSAGES,
        max_estimated_chars: int = MAX_ESTIMATED_CHARS,
) -> list[dict[str, Any]]:
    """
    Build the input window for the model with a bounded context.

    Rules (in priority order):
      1. ``system_prompt`` and ``warning`` are always kept. Client-supplied ``system`` messages are dropped.
         If the server content alone does not fit — ``ContextOverflow`` is raised, even for empty history.
      2. The earliest available turn is kept as the best available context (usually the task description,
         but the code does not assume it).
      3. The last turn is always kept — it is the current request; without it a reply would be meaningless.
         If it does not fit — overflow.
      4. Remaining turns are added newest-to-oldest until either ``max_messages`` or ``max_estimated_chars`` is hit.
         Iteration stops on the first turn that does not fit (``break``, not ``continue``):
         we keep a *continuous* recent tail. Skipping an oversized middle turn would create a gap in the reasoning chain,
         which is worse for multi-step tasks than losing older context.
      5. Oversized ``tool`` outputs are trimmed head + tail before grouping. User/assistant messages are kept whole.

    Pinning: only turn 0 and the last turn are pinned — this is a server-side decision. The client cannot
    influence pinning (a hostile client must not be able to pin everything and force a ``ContextOverflow``).

    Client history cannot contain ``system`` messages, and any ``pinned``
    flags are ignored: pinning is a server concern, not a client one.
    """
    # Step 0. Server-controlled part must fit on its own
    window: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if warning:
        window.append({"role": "system", "content": warning})

    base_chars = sum(_size(m) for m in window)
    if base_chars + RESERVED_OUTPUT_CHARS > max_estimated_chars:
        # Fixes reviewer #3: validate even before checking history.
        raise ContextOverflow("system prompt alone exceeds budget")

    # Step 1. sanitize history
    # Drop client system messages; trim only tool outputs (see _truncate_tool).
    messages = [_truncate_tool(m) for m in history if m.get("role") != "system"]
    if not messages:
        return window

    turns = _group_turns(messages)

    # Step 2. Pin the earliest and the last turn
    # Server-side pinning only. A client ``pinned`` flag is ignored.
    pinned_idx: set[int] = {0, len(turns) - 1}

    pinned_chars = sum(_size(m) for i in pinned_idx for m in turns[i])
    pinned_msgs = sum(len(turns[i]) for i in pinned_idx)

    if pinned_msgs > max_messages:
        raise ContextOverflow("pinned turns exceed message limit")

    if base_chars + pinned_chars + RESERVED_OUTPUT_CHARS > max_estimated_chars:
        # Includes the "current request alone is too large" case —
        # see reviewer #4: this is a deliberate, documented contract.
        raise ContextOverflow("mandatory turns exceed budget")

    budget = (max_estimated_chars - base_chars - RESERVED_OUTPUT_CHARS - pinned_chars)
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
    for i in sorted(keep):
        window.extend(turns[i])

    return window
