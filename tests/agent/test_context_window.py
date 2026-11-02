from typing import Any

import pytest

from services.agent.context.context_window import (
    # public
    AgentContextWindow,
    ContextOverflow,
    build_context,
    # private
    _build_window,
    _group_turns,
    _prepare_context,
    _size,
    _truncate_tool,
    # constants used by tests
    MULTIMODAL_PART_WEIGHT,
    PER_TOOL_MSG_LIMIT,
    RESERVED_OUTPUT_CHARS,
)

SYSTEM = "You are a helpful agent."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def user(content: Any = "hi") -> dict[str, Any]:
    return {"role": "user", "content": content}


def assistant(content: Any = "ok") -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def tool(content: Any = "result") -> dict[str, Any]:
    return {"role": "tool", "content": content}


def system(content: str = "sys") -> dict[str, Any]:
    return {"role": "system", "content": content}


def make_window(
        history: list[dict[str, Any]],
        *,
        system_prompt: str = SYSTEM,
        warning: str | None = None,
        max_estimated_chars: int | None = None,
        max_messages: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build a window from ``history`` using ``_prepare_context`` + ``_build_window``.

    ``_prepare_context`` does not expose ``max_estimated_chars``; it derives the
    history budget from the module-level ``MAX_ESTIMATED_CHARS`` and the effective
    system prompt length. When a caller supplies ``max_estimated_chars``, the
    history budget is recomputed manually as
    ``max_estimated_chars - len(effective_system) - RESERVED_OUTPUT_CHARS`` so
    tests can exercise the budget logic without touching module constants.
    """
    effective_system, history_budget = _prepare_context(system_prompt, warning)

    if max_estimated_chars is not None:
        history_budget = (
                max_estimated_chars - len(effective_system) - RESERVED_OUTPUT_CHARS
        )

    build_kwargs: dict[str, Any] = {"history_budget_chars": history_budget}

    if max_messages is not None:
        build_kwargs["max_messages"] = max_messages

    return _build_window(history, **build_kwargs)


# ---------------------------------------------------------------------------
# Basic: empty history
# ---------------------------------------------------------------------------

def test_empty_history_returns_system_only():
    """An empty history yields the system prompt unchanged and no messages.

    This covers both the "no warning" and "no history" aspects that used to
    live in two identical tests: without a warning the effective system is
    exactly the input prompt, and the window is empty.
    """
    effective, budget = _prepare_context(SYSTEM, None)
    assert effective == SYSTEM
    assert _build_window([], history_budget_chars=budget) == []


def test_empty_history_with_warning():
    effective, budget = _prepare_context(SYSTEM, "careful")
    assert "helpful" in effective
    assert "careful" in effective
    assert _build_window([], history_budget_chars=budget) == []


# ---------------------------------------------------------------------------
# Basic: passthrough and system prompt handling
# ---------------------------------------------------------------------------

def test_small_history_passthrough():
    history = [user("q"), assistant("a")]
    window = make_window(history)
    assert window == history


def test_client_system_messages_dropped():
    history = [system("evil prompt"), user("q"), assistant("a")]
    window = make_window(history)
    assert all(m["content"] != "evil prompt" for m in window)


def test_chronological_order_preserved():
    history = [user(f"q{i}") for i in range(5)]
    window = make_window(history)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert contents == [f"q{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Turn grouping
# ---------------------------------------------------------------------------

def test_tool_belongs_to_previous_turn():
    msgs = [user("q"), assistant("call"), tool("out"), user("next")]
    turns = _group_turns(msgs)
    assert len(turns) == 2
    assert [m["role"] for m in turns[0]] == ["user", "assistant", "tool"]
    assert turns[1] == [user("next")]


def test_group_turns_first_message_not_user():
    turns = _group_turns([assistant("hi"), user("q")])
    assert len(turns) == 2
    assert turns[0] == [assistant("hi")]


def test_tool_only_history_is_one_turn():
    turns = _group_turns([tool("a"), tool("b")])
    assert len(turns) == 1
    assert len(turns[0]) == 2


# ---------------------------------------------------------------------------
# Message limit — exact content
# ---------------------------------------------------------------------------

def test_message_limit_keeps_exact_tail():
    """Only the continuous tail + pinned turns are kept — verify exact content.

    Turn 0 is pinned, turn 9 is pinned (last). ``max_messages=6`` leaves
    room for exactly one more turn — it must be turn 8 (adjacent to the
    last turn), never an older small one. Fixes the previous version,
    which only checked an upper bound.
    """
    history: list[dict[str, Any]] = []
    for i in range(10):
        history.append(user(f"q{i}"))
        history.append(assistant(f"a{i}"))

    window = make_window(history, max_messages=6)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert contents == ["q0", "a0", "q8", "a8", "q9", "a9"]


def test_message_limit_counts_messages_not_turns():
    """Regression: turn count used to be compared against `max_messages`."""
    history: list[dict[str, Any]] = []
    for i in range(20):
        history.append(user(f"q{i}"))
        history.append(assistant(f"a{i}"))

    window = make_window(history, max_messages=10)
    history_msgs = [m for m in window if m["role"] != "system"]
    assert len(history_msgs) <= 10


def test_single_turn_pinned_once():
    history = [user("q"), assistant("a")]
    window = make_window(history, max_messages=3)
    assert window == history


# ---------------------------------------------------------------------------
# Character budget — whole turns only, exact pairs
# ---------------------------------------------------------------------------

def test_char_budget_keeps_whole_turns():
    """Char budget is enforced by dropping whole turns, not partial ones.

    Fixes the previous version, which checked only parity of the message
    count. Every kept user is paired with its own assistant reply, and
    the paired messages come from the same turn index.
    """
    history: list[dict[str, Any]] = []
    for i in range(10):
        history.append(user(f"q{i}-" + "x" * 500))
        history.append(assistant(f"a{i}-" + "x" * 500))

    max_estimated_chars = 15_000
    window = make_window(history, max_estimated_chars=max_estimated_chars)
    effective, _ = _prepare_context(SYSTEM, None)

    total = sum(_size(m) for m in window)
    assert total + len(effective) + RESERVED_OUTPUT_CHARS <= max_estimated_chars

    contents = [m["content"] for m in window if m["role"] != "system"]
    assert len(contents) % 2 == 0
    for i in range(0, len(contents), 2):
        assert contents[i].startswith("q")
        assert contents[i + 1].startswith("a")
        # Pair must be from the same turn: q3 ↔ a3, never q3 ↔ a5.
        assert contents[i].split("-")[0][1:] == contents[i + 1].split("-")[0][1:]


def test_reserved_output_chars_reduces_budget():
    """``max_estimated_chars=15_000`` leaves less than ``RESERVED_OUTPUT_CHARS`` for history.

    Two 5000-char messages need 10_000 chars of budget; at 15_000 the
    leftover after system + reserve is too small → overflow. At 25_000
    both messages fit.
    """
    history = [user("x" * 5000), assistant("x" * 5000)]

    with pytest.raises(ContextOverflow):
        make_window(history, max_estimated_chars=15_000)

    window = make_window(history, max_estimated_chars=25_000)
    assert len(window) == 2


# ---------------------------------------------------------------------------
# Pinning: first + last turn fully; client flag ignored
# ---------------------------------------------------------------------------

def test_first_and_last_turn_fully_kept():
    """Turn 0 and the last turn are kept whole — both user and assistant.

    Fixes the previous version, which only looked for the `q*` prefix and
    would pass even if the matching assistant reply was silently dropped.
    """
    history: list[dict[str, Any]] = []
    for i in range(20):
        history.append(user(f"q{i}" + "x" * 500))
        history.append(assistant(f"a{i}" + "x" * 500))

    window = make_window(history, max_estimated_chars=20_000)
    contents = [m["content"] for m in window if m["role"] != "system"]

    assert "q0" + "x" * 500 in contents
    assert "a0" + "x" * 500 in contents
    assert "q19" + "x" * 500 in contents
    assert "a19" + "x" * 500 in contents


def test_client_pinned_flag_ignored():
    """A client `pinned` flag has no effect: pinning is server-side only."""
    history = [
        user("q0"), assistant("a0"),
        {"role": "user", "content": "q1", "pinned": True},
        assistant("a1"),
        user("q2"), assistant("a2"),
    ]
    window = make_window(history, max_messages=5)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert "q1" not in contents
    assert "q0" in contents
    assert "q2" in contents


# ---------------------------------------------------------------------------
# Continuous tail (break, not continue)
# ---------------------------------------------------------------------------

def test_continuous_tail_break_on_oversized_turn():
    """A huge turn stops the walk; older small turns are NOT skipped in."""
    history = [
        user("q0"), assistant("a0"),
        user("small"), assistant("small"),
        user("x" * 20_000), assistant("x" * 20_000),
        user("qN"), assistant("aN"),
    ]
    window = make_window(history, max_estimated_chars=20_000)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert "small" not in contents
    assert "q0" in contents
    assert "qN" in contents


# ---------------------------------------------------------------------------
# ContextOverflow: predictable failures
# ---------------------------------------------------------------------------

def test_overflow_system_alone_too_big():
    # ``_prepare_context`` derives the budget from the module-level
    # ``MAX_ESTIMATED_CHARS`` and does not accept a per-call
    # ``max_estimated_chars``, so the overflow path is exercised with a
    # system prompt that is too large for any realistic configuration.
    with pytest.raises(ContextOverflow):
        _prepare_context("x" * 10_000_000, None)


def test_overflow_warning_alone_too_big():
    """A warning participates in the effective system length — overflow with empty history.

    Regression: previously only the system prompt was checked directly;
    the fact that ``warning`` also participates in the length calculation was
    only implicit. The value is deliberately oversized because the budget
    comes from the module-level ``MAX_ESTIMATED_CHARS`` and cannot be injected
    through ``_prepare_context``.
    """
    with pytest.raises(ContextOverflow):
        _prepare_context("small", "w" * 10_000_000)


def test_overflow_pinned_msgs_exceed_max_messages():
    history = [user("q"), assistant("a")]
    with pytest.raises(ContextOverflow):
        make_window(history, max_messages=1)


def test_overflow_current_request_too_big():
    """Explicit case: last turn alone exceeds the budget."""
    history = [user("q0"), assistant("a0"), user("x" * 100_000)]
    with pytest.raises(ContextOverflow):
        make_window(history, max_estimated_chars=20_000)


def test_overflow_current_request_multi_message():
    """A multi-message last turn exceeding the budget also overflows.

    Fixes the previous version, which used a single-message last turn and
    therefore only accidentally matched the contract.
    """
    history = [
        user("q0"), assistant("a0"),
        user("q1"), assistant("a1"),
        user("x" * 15_000), assistant("y" * 15_000),
    ]
    with pytest.raises(ContextOverflow):
        make_window(history, max_estimated_chars=20_000)


def test_overflow_max_messages_zero_with_history():
    """``max_messages=0`` with any history overflows — nothing can be pinned."""
    with pytest.raises(ContextOverflow):
        make_window([user("q")], max_messages=0)


# ---------------------------------------------------------------------------
# Tool truncation
# ---------------------------------------------------------------------------

def test_tool_truncation_preserves_head_and_tail():
    content = "start-marker" + "x" * 10_000 + "end-marker"
    result = _truncate_tool(tool(content))
    assert result["content"].startswith("start-marker")
    assert result["content"].endswith("end-marker")
    assert "truncated" in result["content"]
    assert _size(result) < _size(tool(content))


def test_tool_truncation_marker_reports_correct_count():
    """The marker reports the exact removed count, and the total size is exact.

    Note: the implementation adds the marker on top of ``head + tail``,
    so the result is *slightly* over ``PER_TOOL_MSG_LIMIT`` by design.
    The test pins the exact size instead of pretending a strict ``≤``.
    """
    content = "y" * 10_000
    result = _truncate_tool(tool(content))

    head = PER_TOOL_MSG_LIMIT * 2 // 3
    tail = PER_TOOL_MSG_LIMIT // 3
    removed = len(content) - head - tail

    assert f"[truncated {removed} chars]" in result["content"]
    # head + tail + "\n" + "[truncated N chars]" + "\n" + two ellipses
    expected = head + tail + len(f"[truncated {removed} chars]") + 2 + 2
    assert _size(result) == expected


def test_tool_at_limit_unchanged():
    """A tool message exactly at ``PER_TOOL_MSG_LIMIT`` is not touched."""
    msg = tool("x" * PER_TOOL_MSG_LIMIT)
    assert _truncate_tool(msg) is msg


def test_tool_over_limit_by_one_truncated():
    """A tool message one char over ``PER_TOOL_MSG_LIMIT`` is truncated."""
    msg = tool("x" * (PER_TOOL_MSG_LIMIT + 1))
    result = _truncate_tool(msg)
    assert result is not msg
    assert "truncated" in result["content"]


def test_tool_short_message_unchanged():
    msg = tool("small")
    assert _truncate_tool(msg) is msg


def test_user_message_not_truncated():
    msg = user("x" * 10_000)
    assert _truncate_tool(msg) is msg


def test_assistant_message_not_truncated():
    msg = assistant("y" * 10_000)
    assert _truncate_tool(msg) is msg


def test_truncate_tool_multimodal_not_truncated():
    msg = tool([{"type": "text", "text": "x" * 10_000}])
    assert _truncate_tool(msg) is msg


# ---------------------------------------------------------------------------
# `_size`: characters and multimodal weighting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content,expected", [
    ("hello", 5),
    ("привет", 6),
    ("", 0),
    (None, 0),
    ([], 0),
    ([{"type": "text", "text": "hi"}], 2),
    ([{"type": "image_url", "image_url": {"url": "..."}}], MULTIMODAL_PART_WEIGHT),
    ([{"type": "text", "text": "hi"},
      {"type": "image_url", "image_url": {"url": "..."}}],
     2 + MULTIMODAL_PART_WEIGHT),
    ([{"type": "text", "text": "hi"}, "raw-string-part"], 2),
])
def test_size(content, expected):
    assert _size({"role": "user", "content": content}) == expected


# ---------------------------------------------------------------------------
# Multimodal budget: boundary case
# ---------------------------------------------------------------------------

def test_multimodal_budget_boundary():
    """Boundary: exactly one image part fits, two do not.

    Fixes the previous version, which used a budget with a wide margin
    and did not prove that the multimodal weight itself participates in
    the calculation. Here the budget is chosen so that a single part
    hits the limit exactly.
    """
    one = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "..."}}],
    }
    two = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "..."}}] * 2,
    }
    effective, _ = _prepare_context(SYSTEM, None)
    boundary = (
            len(effective) + RESERVED_OUTPUT_CHARS + MULTIMODAL_PART_WEIGHT
    )

    # Exactly one image fits — no overflow.
    window = make_window([one], max_estimated_chars=boundary)
    assert window[-1] == one

    # Two images overflow by exactly MULTIMODAL_PART_WEIGHT.
    with pytest.raises(ContextOverflow):
        make_window([two], max_estimated_chars=boundary)


# ---------------------------------------------------------------------------
# Immutability of caller's history
# ---------------------------------------------------------------------------

def test_history_not_mutated():
    """``_build_window`` must not mutate the caller's list or its messages.

    ``_truncate_tool`` returns a new dict via ``{**msg, ...}``, and
    ``_build_window`` collects results into a fresh list. This test
    locks that contract in.
    """
    history = [user("q"), tool("x" * 10_000), user("q2")]
    snapshot = [dict(m) for m in history]

    make_window(history)

    assert history == snapshot
    # The original tool message still holds its full content.
    assert history[1]["content"] == "x" * 10_000


# ---------------------------------------------------------------------------
# Unbalanced first turn
# ---------------------------------------------------------------------------

def test_unbalanced_first_turn_kept_whole():
    """History starting with assistant/tool: the first group is pinned whole.

    ``_group_turns`` handles histories that do not start with ``user``;
    this verifies that ``_build_window`` also preserves such a group as a
    complete pinned turn, not just its leading message.
    """
    history = [assistant("hi"), tool("out"), user("q1"), assistant("a1")]
    window = make_window(history, max_estimated_chars=15_000)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert "hi" in contents
    assert "out" in contents
    assert "q1" in contents
    assert "a1" in contents


# ---------------------------------------------------------------------------
# Public API: build_context
# ---------------------------------------------------------------------------

def test_build_context_returns_agent_context_window():
    result = build_context("sys", None, [])
    assert isinstance(result, AgentContextWindow)
    assert result.system == "sys"
    assert result.messages == []


def test_build_context_appends_warning():
    without = build_context("sys", None, [])
    with_warning = build_context("sys", "WARNING: budget", [])
    assert without.system == "sys"
    assert with_warning.system == "sys\n\nWARNING: budget"


def test_build_context_drops_client_system_role():
    result = build_context("sys", None, [
        system("injected"),
        user("hi"),
    ])
    assert all(m["role"] != "system" for m in result.messages)
    assert result.messages == [user("hi")]


def test_build_context_propagates_system_overflow(monkeypatch):
    """Trigger overflow in ``_prepare_context`` via monkeypatch on the module constant.

    ``build_context`` does not accept ``max_estimated_chars``, and passing
    a huge ``system_prompt`` would be less readable than a one-line
    monkeypatch on ``MAX_ESTIMATED_CHARS``.
    """
    import services.agent.context.context_window as cw_mod
    monkeypatch.setattr(cw_mod, "MAX_ESTIMATED_CHARS", 100)

    with pytest.raises(ContextOverflow):
        build_context("sys", None, [])


def test_build_context_propagates_pinned_overflow():
    """Four pinned messages exceed ``max_messages=2`` — overflow from the message limit."""
    history = [
        user("q0"), assistant("a0"),
        user("q1"), assistant("a1"),
    ]

    with pytest.raises(ContextOverflow):
        build_context("sys", None, history, max_messages=2)


def test_build_context_empty_history():
    """Separate from ``test_build_context_returns_agent_context_window``.

    That test pins the return type; this one pins the no-overflow contract
    on empty history.
    """
    result = build_context("sys", None, [])
    assert result == AgentContextWindow(system="sys", messages=[])


def test_build_context_max_messages_exactly_pinned():
    """Boundary: max_messages == pinned_msgs is allowed, not overflow.

    ``_build_window`` uses ``>``, so exactly-equal must pass. This test pins
    the boundary — a switch to ``>=`` would break it.
    """
    history = [user("q0"), assistant("a0")]  # one turn, two messages, both pinned

    result = build_context("sys", None, history, max_messages=2)

    assert result.messages == history


def test_build_context_max_messages_truncates():
    """max_messages truncates while keeping the pinned first and last turns.

    Char-budget here has huge slack (MAX_ESTIMATED_CHARS = 60_000 by default,
    the whole history is ~16 characters: 8 messages × 2 chars), so truncation
    is guaranteed to come from max_messages, not from the character budget.
    This test therefore proves that ``max_messages`` really reaches
    ``_build_window``, not just that it is validated.
    """
    history = [
        user("u1"), assistant("a1"),
        user("u2"), assistant("a2"),
        user("u3"), assistant("a3"),
        user("u4"), assistant("a4"),
    ]

    result = build_context("sys", None, history, max_messages=6)

    assert len(result.messages) == 6
    # Pinned turn 0 (u1, a1) and pinned last turn (u4, a4) survive.
    assert result.messages[0]["content"] == "u1"
    assert result.messages[1]["content"] == "a1"
    assert result.messages[-2]["content"] == "u4"
    assert result.messages[-1]["content"] == "a4"
    # Turn 1 (u2, a2) is dropped: walking backwards from turn 3, only turn 2
    # fits into the 2-message budget left after pinning, then the message
    # limit stops the walk.
    assert {"role": "user", "content": "u2"} not in result.messages
    assert {"role": "assistant", "content": "a2"} not in result.messages
    # Turn 2 (u3, a3) is kept as the continuous tail adjacent to the last turn.
    assert {"role": "user", "content": "u3"} in result.messages
    assert {"role": "assistant", "content": "a3"} in result.messages
