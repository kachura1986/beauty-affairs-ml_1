"""EXECUTE stage tests: execute_node, pairing invariant, regressions."""
import pytest

from services.agent.agent_graph.nodes import execute_node
from services.agent.domain import AgentPolicy, TerminationReason
from services.agent.tools import Tool, ToolRegistry

from _testing import EchoArgs, FakeRuntime, failing_tool, timeout_tool


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_execute_successful_tool_call(trace, make_context, make_state):
    """A successful call clears pending calls and appends a tool message."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    ctx = make_context()
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["pending_tool_calls"] == []
    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert "hi" in result["history"][0]["content"]
    assert result["action_counts"]


async def test_execute_empty_batch_is_noop(trace, make_context, make_state):
    """An empty pending_tool_calls produces no history and no termination."""
    ctx = make_context()
    state = make_state(ctx)
    state["pending_tool_calls"] = []

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["history"] == []
    assert result["pending_tool_calls"] == []
    assert "termination_reason" not in result


# ---------------------------------------------------------------------------
# Failure handling: irrecoverable, temporary, skipped, repeated
# ---------------------------------------------------------------------------

async def test_execute_irrecoverable_failed_action_skipped(trace, make_context, make_state, make_registry):
    """A non-retryable failure is recorded, then skipped on the next attempt as a paired tool message (not an observation)."""
    call = {"id": "c1", "name": "failing", "arguments": {}}
    registry = make_registry(failing_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)

    # First: fail and record fingerprint
    state["pending_tool_calls"] = [call]
    first = await execute_node(state, FakeRuntime(ctx))
    assert first["failed_actions"]

    # Second: same call must be answered synthetically with a tool message
    state["pending_tool_calls"] = [call]
    state["failed_actions"] = first["failed_actions"]
    second = await execute_node(state, FakeRuntime(ctx))

    assert len(second["history"]) == 1
    assert second["history"][0]["role"] == "tool"
    assert second["history"][0]["tool_call_id"] == "c1"
    assert "previously failed irrecoverably" in second["history"][0]["content"]


async def test_execute_repeated_action_terminates(trace, make_context, make_state):
    """Exceeding repeat_limit terminates with REPEATED_ACTION while still emitting a paired tool message (no `break`)."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(policy=AgentPolicy(empty_limit=3, repeat_limit=2))
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]
    state["action_counts"] = {
        ToolRegistry.fingerprint(call): 2
    }

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.REPEATED_ACTION
    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert result["history"][0]["tool_call_id"] == "c1"
    assert "too many times" in result["history"][0]["content"]


async def test_execute_temporary_failure_does_not_add_to_failed_set(trace, make_context, make_state, make_registry):
    """Temporary failures stay retryable and are not added to failed_actions."""
    call = {"id": "c1", "name": "timeout", "arguments": {}}
    registry = make_registry(timeout_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["failed_actions"] == []


async def test_execute_skips_duplicate_failed_action(trace, make_context, make_state, make_registry):
    """A call whose fingerprint is already in failed_actions is skipped as a paired tool message."""
    call = {"id": "c1", "name": "failing", "arguments": {}}
    registry = make_registry(failing_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)
    fp = ToolRegistry.fingerprint(call)
    state["failed_actions"] = [fp]
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert result["history"][0]["tool_call_id"] == "c1"
    assert "previously failed irrecoverably" in result["history"][0]["content"]


async def test_execute_multiple_calls_second_fails_irrecoverable(trace, make_context, make_state, make_registry):
    """A second call failing irrecoverably still records both results."""
    ok = {"id": "c1", "name": "echo", "arguments": {"value": "ok"}}
    bad = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(failing_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [ok, bad]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["failed_actions"]) == 1
    assert len(result["history"]) == 2


async def test_execute_multiple_tool_calls_partial_failure(trace, make_context, make_state, make_registry):
    """When one of several calls fails, all results are still appended to history."""
    ok = {"id": "c1", "name": "echo", "arguments": {"value": "ok"}}
    fail = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(failing_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [ok, fail]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["history"]) == 2
    assert result["history"][0]["role"] == "tool"
    assert result["history"][1]["role"] == "tool"
    assert result["failed_actions"]


# ---------------------------------------------------------------------------
# Budget: preflight and mid-batch
# ---------------------------------------------------------------------------

async def test_execute_preflight_time_skips_tool(trace, make_context, make_state):
    """An already-expired wall-clock budget short-circuits before invoke() while still emitting a paired tool message."""
    import time

    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(max_seconds=0.001)
    state = make_state(ctx)
    state["budget"].started = time.monotonic() - 10
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TIME
    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert result["history"][0]["tool_call_id"] == "c1"
    assert "was not executed" in result["history"][0]["content"]


async def test_execute_stops_after_budget_exceeded_between_calls(trace, make_context, make_state, make_registry):
    """If the budget expires during the first tool, the second must not execute."""
    calls_made = []

    async def slow_fn(**kwargs):
        calls_made.append(1)
        state["budget"].started -= 10_000  # emulate time passing
        return "ok"

    registry = make_registry(
        Tool(name="slow", description="", fn=slow_fn, args_model=EchoArgs)
    )
    ctx = make_context(tools=registry, max_seconds=10)
    state = make_state(ctx)
    call1 = {"id": "c1", "name": "slow", "arguments": {}}
    call2 = {"id": "c2", "name": "echo", "arguments": {}}
    state["pending_tool_calls"] = [call1, call2]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(calls_made) == 1
    assert result["termination_reason"] == TerminationReason.BUDGET_TIME


# ---------------------------------------------------------------------------
# Pairing invariant: every pending call gets exactly one tool message
# ---------------------------------------------------------------------------


@pytest.mark.invariant
@pytest.mark.parametrize(
    "calls, policy_kwargs, extra_tools",
    [
        # Happy path
        (
                [
                    {"id": "c1", "name": "echo", "arguments": {"value": "a"}},
                    {"id": "c2", "name": "echo", "arguments": {"value": "b"}},
                ],
                {"empty_limit": 3, "repeat_limit": 3},
                [],
        ),
        # failed_set: both calls share a fingerprint; first fails, second skipped
        (
                [
                    {"id": "c1", "name": "failing", "arguments": {}},
                    {"id": "c2", "name": "failing", "arguments": {}},
                ],
                {"empty_limit": 3, "repeat_limit": 3},
                [failing_tool()],
        ),
        # REPEATED_ACTION trips mid-batch, last call is skipped
        (
                [
                    {"id": "c1", "name": "echo", "arguments": {"value": "x"}},
                    {"id": "c2", "name": "echo", "arguments": {"value": "x"}},
                    {"id": "c3", "name": "echo", "arguments": {"value": "x"}},
                ],
                {"empty_limit": 3, "repeat_limit": 1},
                [],
        ),
    ],
)
async def test_property_pending_calls_are_all_paired(
        trace, make_context, make_state, make_registry, calls, policy_kwargs, extra_tools,
):
    """Every pending call receives exactly one tool message with a matching id on any exit path — no `break`, no early return, no observation in place of a tool message."""
    registry = make_registry(*extra_tools)
    ctx = make_context(tools=registry, policy=AgentPolicy(**policy_kwargs))
    state = make_state(ctx)
    state["pending_tool_calls"] = list(calls)

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["history"]) == len(calls)
    assert all(m["role"] == "tool" for m in result["history"])
    assert [m["tool_call_id"] for m in result["history"]] == [c["id"] for c in calls]


# ---------------------------------------------------------------------------
# Regression tests for the execute_node tool_call pairing invariant
# ---------------------------------------------------------------------------

@pytest.mark.regression
async def test_regression_failed_set_same_batch(trace, make_context, make_state, make_registry):
    """Two identical failed calls in one batch each get a matching tool message — the second was once answered with an observation."""
    call1 = {"id": "c1", "name": "failing", "arguments": {}}
    call2 = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(failing_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [call1, call2]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result.get("termination_reason") is None
    assert len(result["failed_actions"]) == 1
    assert len(result["history"]) == 2
    assert [m["role"] for m in result["history"]] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in result["history"]] == ["c1", "c2"]
    # First ran and failed; second was answered synthetically.
    assert "internal error" in result["history"][0]["content"]
    assert "previously failed irrecoverably" in result["history"][1]["content"]


@pytest.mark.regression
async def test_regression_repeated_action_three_calls(trace, make_context, make_state):
    """Three identical calls with repeat_limit=1 produce three paired tool messages — a `break` on trip left the tail unpaired."""
    call1 = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    call2 = {"id": "c2", "name": "echo", "arguments": {"value": "x"}}
    call3 = {"id": "c3", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(policy=AgentPolicy(empty_limit=3, repeat_limit=1))
    state = make_state(ctx)
    state["pending_tool_calls"] = [call1, call2, call3]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.REPEATED_ACTION
    assert len(result["history"]) == 3
    assert [m["role"] for m in result["history"]] == ["tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in result["history"]] == ["c1", "c2", "c3"]

    # First: real result. Second: repeated_action. Third: skipped_due_to_termination.
    assert "x" in result["history"][0]["content"]
    assert "too many times" in result["history"][1]["content"]
    assert "was not executed" in result["history"][2]["content"]

    # action_counts incremented once per identical attempt that reached branch (3):
    # call1 -> 1, call2 -> 2 (trip limit), call3 -> skipped before counting.
    fp = ToolRegistry.fingerprint(call1)
    assert result["action_counts"] == {fp: 2}


@pytest.mark.regression
async def test_regression_budget_exceeded_mid_batch(trace, make_context, make_state, make_registry):
    """Budget overrun after the first call leaves the trailing calls paired with skipped_due_to_termination results."""
    call1 = {"id": "c1", "name": "slow", "arguments": {}}
    call2 = {"id": "c2", "name": "echo", "arguments": {}}
    call3 = {"id": "c3", "name": "echo", "arguments": {}}

    async def slow_fn(**kwargs):
        # Emulate the first call burning through the wall-clock budget.
        state["budget"].started -= 10_000
        return "ok"

    registry = make_registry(
        Tool(name="slow", description="", fn=slow_fn, args_model=EchoArgs)
    )
    ctx = make_context(tools=registry, max_seconds=10)
    state = make_state(ctx)
    state["pending_tool_calls"] = [call1, call2, call3]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TIME
    assert len(result["history"]) == 3
    assert [m["role"] for m in result["history"]] == ["tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in result["history"]] == ["c1", "c2", "c3"]

    # First: real result. Trailing two: skipped with the reason spelled out.
    assert "ok" in result["history"][0]["content"]
    for skipped in result["history"][1:]:
        assert "was not executed" in skipped["content"]
        assert "budget_time" in skipped["content"]
        assert "ERROR execution_failed" in skipped["content"]


@pytest.mark.regression
async def test_regression_skipped_message_carries_reason(trace, make_context, make_state):
    """Skipped calls embed the termination reason — the tripping call carries the repeated_action flavour, not the skip flavour."""
    call1 = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    call2 = {"id": "c2", "name": "echo", "arguments": {"value": "x"}}
    call3 = {"id": "c3", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(policy=AgentPolicy(empty_limit=3, repeat_limit=1))
    state = make_state(ctx)
    state["pending_tool_calls"] = [call1, call2, call3]

    result = await execute_node(state, FakeRuntime(ctx))

    # c2 trips the limit; its message is the repeat-limit flavour, not skipped.
    repeated = result["history"][1]
    assert repeated["tool_call_id"] == "c2"
    assert "too many times" in repeated["content"]
    assert "repeated_action" not in repeated["content"]

    # c3 goes through branch (1) and embeds the termination reason.
    skipped = result["history"][2]
    assert skipped["tool_call_id"] == "c3"
    assert "repeated_action" in skipped["content"]
    assert "was not executed" in skipped["content"]


# ---------------------------------------------------------------------------
# last_tool_ok: explicit tracking of the most recent successful tool result
# ---------------------------------------------------------------------------

async def test_execute_sets_last_tool_ok_on_success(trace, make_context, make_state):
    """A successful tool call records its rendered result in last_tool_ok — the explicit source used by finalize's deterministic fallback."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    ctx = make_context()
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["last_tool_ok"] is not None
    assert "hi" in result["last_tool_ok"]


async def test_execute_keeps_last_tool_ok_on_subsequent_error(trace, make_context, make_state, make_registry):
    """A failed round must not overwrite an earlier successful result — otherwise the fallback degrades to an error explanation instead of the useful data."""
    ok_call = {"id": "c1", "name": "echo", "arguments": {"value": "saved"}}
    bad_call = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(failing_tool())
    ctx = make_context(tools=registry)
    state = make_state(ctx)

    # Round 1: successful call records last_tool_ok.
    state["pending_tool_calls"] = [ok_call]
    first = await execute_node(state, FakeRuntime(ctx))
    assert first["last_tool_ok"] is not None
    assert "saved" in first["last_tool_ok"]

    # Round 2: only errors — last_tool_ok must carry over unchanged.
    state["pending_tool_calls"] = [bad_call]
    state["last_tool_ok"] = first["last_tool_ok"]
    second = await execute_node(state, FakeRuntime(ctx))

    assert second["last_tool_ok"] == first["last_tool_ok"]


async def test_execute_handles_missing_last_tool_ok_key(trace, make_context, make_state):
    """execute_node tolerates a state without 'last_tool_ok' (total=False) instead of raising KeyError."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    ctx = make_context()
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]
    del state["last_tool_ok"]  # explicitly simulate the missing key

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["last_tool_ok"] is not None
    assert "hi" in result["last_tool_ok"]
