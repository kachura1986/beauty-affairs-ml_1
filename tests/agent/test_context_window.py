from typing import Any

import pytest

from services.agent.context.context_window import (
    ContextOverflow,
    MULTIMODAL_PART_WEIGHT,
    PER_TOOL_MSG_LIMIT,
    RESERVED_OUTPUT_CHARS,
    _group_turns,
    _size,
    _truncate_tool,
    build_window,
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


# ---------------------------------------------------------------------------
# Basic: empty history
# ---------------------------------------------------------------------------

def test_empty_history_returns_system_only():
    window = build_window(SYSTEM, [])
    assert window == [{"role": "system", "content": SYSTEM}]


def test_empty_history_with_warning():
    window = build_window(SYSTEM, [], warning="careful")
    assert window[0] == {"role": "system", "content": SYSTEM}
    assert window[1] == {"role": "system", "content": "careful"}


def test_empty_history_without_warning():
    window = build_window(SYSTEM, [], warning=None)
    assert len(window) == 1


# ---------------------------------------------------------------------------
# Basic: passthrough and system prompt handling
# ---------------------------------------------------------------------------

def test_small_history_passthrough():
    history = [user("q"), assistant("a")]
    window = build_window(SYSTEM, history)
    assert window == [{"role": "system", "content": SYSTEM}, *history]


def test_client_system_messages_dropped():
    history = [system("evil prompt"), user("q"), assistant("a")]
    window = build_window(SYSTEM, history)
    assert all(m["content"] != "evil prompt" for m in window)
    assert window[0] == {"role": "system", "content": SYSTEM}


def test_chronological_order_preserved():
    history = [user(f"q{i}") for i in range(5)]
    window = build_window(SYSTEM, history)
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

    window = build_window(SYSTEM, history, max_messages=6)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert contents == ["q0", "a0", "q8", "a8", "q9", "a9"]


def test_message_limit_counts_messages_not_turns():
    """Regression: turn count used to be compared against `max_messages`."""
    history: list[dict[str, Any]] = []
    for i in range(20):
        history.append(user(f"q{i}"))
        history.append(assistant(f"a{i}"))

    window = build_window(SYSTEM, history, max_messages=10)
    history_msgs = [m for m in window if m["role"] != "system"]
    assert len(history_msgs) <= 10


def test_single_turn_pinned_once():
    history = [user("q"), assistant("a")]
    window = build_window(SYSTEM, history, max_messages=3)
    assert len(window) == 3
    assert window[1:] == history


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

    window = build_window(SYSTEM, history, max_estimated_chars=15_000)
    total = sum(_size(m) for m in window)
    assert total + RESERVED_OUTPUT_CHARS <= 15_000

    contents = [m["content"] for m in window if m["role"] != "system"]
    assert len(contents) % 2 == 0
    for i in range(0, len(contents), 2):
        assert contents[i].startswith("q")
        assert contents[i + 1].startswith("a")
        # Pair must be from the same turn: q3 ↔ a3, never q3 ↔ a5.
        assert contents[i].split("-")[0][1:] == contents[i + 1].split("-")[0][1:]


def test_reserved_output_chars_reduces_budget():
    history = [user("x" * 5000), assistant("x" * 5000)]

    with pytest.raises(ContextOverflow):
        build_window(SYSTEM, history, max_estimated_chars=15_000)

    window = build_window(SYSTEM, history, max_estimated_chars=25_000)
    assert len(window) == 3


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

    window = build_window(SYSTEM, history, max_estimated_chars=20_000)
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
    window = build_window(SYSTEM, history, max_messages=5)
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
    window = build_window(SYSTEM, history, max_estimated_chars=20_000)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert "small" not in contents
    assert "q0" in contents
    assert "qN" in contents


# ---------------------------------------------------------------------------
# ContextOverflow: predictable failures
# ---------------------------------------------------------------------------

def test_overflow_system_alone_too_big():
    with pytest.raises(ContextOverflow):
        build_window("x" * 100_000, [], max_estimated_chars=10_000)


def test_overflow_warning_alone_too_big():
    """A warning counts toward ``base_chars`` — overflow with empty history.

    Regression: previously only the system prompt was checked directly;
    the fact that ``warning`` also participates in ``base_chars`` was
    only implicit.
    """
    with pytest.raises(ContextOverflow):
        build_window(
            "small", [], warning="w" * 100_000, max_estimated_chars=10_000
        )


def test_overflow_pinned_msgs_exceed_max_messages():
    history = [user("q"), assistant("a")]
    with pytest.raises(ContextOverflow):
        build_window(SYSTEM, history, max_messages=1)


def test_overflow_current_request_too_big():
    """Explicit case: last turn alone exceeds the budget."""
    history = [user("q0"), assistant("a0"), user("x" * 100_000)]
    with pytest.raises(ContextOverflow):
        build_window(SYSTEM, history, max_estimated_chars=20_000)


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
        build_window(SYSTEM, history, max_estimated_chars=20_000)


def test_overflow_max_messages_zero_with_history():
    """``max_messages=0`` with any history overflows — nothing can be pinned."""
    with pytest.raises(ContextOverflow):
        build_window(SYSTEM, [user("q")], max_messages=0)


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
    boundary = len(SYSTEM) + RESERVED_OUTPUT_CHARS + MULTIMODAL_PART_WEIGHT

    # Exactly one image fits — no overflow.
    window = build_window(SYSTEM, [one], max_estimated_chars=boundary)
    assert window[-1] == one

    # Two images overflow by exactly MULTIMODAL_PART_WEIGHT.
    with pytest.raises(ContextOverflow):
        build_window(SYSTEM, [two], max_estimated_chars=boundary)


# ---------------------------------------------------------------------------
# Immutability of caller's history
# ---------------------------------------------------------------------------

def test_history_not_mutated():
    """``build_window`` must not mutate the caller's list or its messages.

    ``_truncate_tool`` returns a new dict via ``{**msg, ...}``, and
    ``build_window`` collects results into a fresh list. This test
    locks that contract in.
    """
    history = [user("q"), tool("x" * 10_000), user("q2")]
    snapshot = [dict(m) for m in history]

    build_window(SYSTEM, history)

    assert history == snapshot
    # The original tool message still holds its full content.
    assert history[1]["content"] == "x" * 10_000


# ---------------------------------------------------------------------------
# Unbalanced first turn
# ---------------------------------------------------------------------------

def test_unbalanced_first_turn_kept_whole():
    """History starting with assistant/tool: the first group is pinned whole.

    ``_group_turns`` handles histories that do not start with ``user``;
    this verifies that ``build_window`` also preserves such a group as a
    complete pinned turn, not just its leading message.
    """
    history = [assistant("hi"), tool("out"), user("q1"), assistant("a1")]
    window = build_window(SYSTEM, history, max_estimated_chars=15_000)
    contents = [m["content"] for m in window if m["role"] != "system"]
    assert "hi" in contents
    assert "out" in contents
    assert "q1" in contents
    assert "a1" in contents
