"""Terminal stage tests: verify_node, finish_node, routing, create_initial_state.

Grouped by graph stage: this file covers everything after EXECUTE — the
VERIFY/FINISH terminal nodes, the routing predicates, and the state factory.
DECIDE lives in test_decide.py, EXECUTE in test_execute.py.
"""
import pytest

from services.agent.agent_graph.nodes import finish_node, verify_node
from services.agent.agent_graph.routing import (
    route_after_decide,
    route_after_execute,
    route_after_verify,
)
from services.agent.agent_graph.state import create_initial_state
from services.agent.domain import (
    AgentResult,
    BudgetLimits,
    Task,
    TerminationReason,
)

from _testing import FakeRuntime


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------

def test_verify_no_postcondition_done_unverified(trace, make_context, make_state):
    """No postcondition configured yields DONE_UNVERIFIED."""
    ctx = make_context(postcondition=None)
    state = make_state(ctx)
    state["pending_answer"] = "answer"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.DONE_UNVERIFIED
    assert result["answer"] == "answer"
    assert result["pending_answer"] is None


def test_verify_postcondition_ok_done(trace, make_context, make_state):
    """A passing postcondition yields DONE."""
    ctx = make_context(postcondition=lambda a: (True, "fine"))
    state = make_state(ctx)
    state["pending_answer"] = "good"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.DONE
    assert result["answer"] == "good"


def test_verify_postcondition_fails_adds_observation(trace, make_context, make_state):
    """A failing postcondition returns to DECIDE with an observation.

    The original answer is not accepted; instead the note from the
    postcondition is fed back to the model for another attempt.
    """
    ctx = make_context(postcondition=lambda a: (False, "missing detail"))
    state = make_state(ctx)
    state["pending_answer"] = "bad"

    result = verify_node(state, FakeRuntime(ctx))

    assert "termination_reason" not in result
    assert result["history"][0]["role"] == "user"
    assert "Postcondition not satisfied" in result["history"][0]["content"]
    assert "missing detail" in result["history"][0]["content"]


def test_verify_postcondition_exception_returns_postcondition_error(trace, make_context, make_state):
    """A raising postcondition preserves the answer and reports POSTCONDITION_ERROR."""

    def broken(_):
        raise RuntimeError("boom")

    ctx = make_context(postcondition=broken)
    state = make_state(ctx)
    state["pending_answer"] = "answer"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.POSTCONDITION_ERROR
    assert result["answer"] == "answer"


def test_verify_postcondition_returns_invalid_shape(trace, make_context, make_state):
    """Returning a non-tuple from postcondition yields POSTCONDITION_ERROR."""
    ctx = make_context(postcondition=lambda a: "not a tuple")
    state = make_state(ctx)
    state["pending_answer"] = "x"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.POSTCONDITION_ERROR
    assert result["answer"] == "x"


def test_verify_without_pending_answer_raises(trace, make_context, make_state):
    """Reaching VERIFY without a pending answer is a contract violation."""
    ctx = make_context()
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="VERIFY phase has no pending answer"):
        verify_node(state, FakeRuntime(ctx))


def test_verify_postcondition_note_in_observation(trace, make_context, make_state):
    """The postcondition note is embedded verbatim in the observation."""
    ctx = make_context(postcondition=lambda a: (False, "note-XYZ"))
    state = make_state(ctx)
    state["pending_answer"] = "x"

    result = verify_node(state, FakeRuntime(ctx))

    assert "note-XYZ" in result["history"][0]["content"]


def test_verify_postcondition_fails_with_empty_note(trace, make_context, make_state):
    """An empty note from postcondition does not break the flow."""
    ctx = make_context(postcondition=lambda a: (False, ""))
    state = make_state(ctx)
    state["pending_answer"] = "x"

    result = verify_node(state, FakeRuntime(ctx))

    assert "Postcondition not satisfied" in result["history"][0]["content"]


# ---------------------------------------------------------------------------
# finish_node
# ---------------------------------------------------------------------------

def test_finish_without_termination_reason_raises(trace, make_context, make_state):
    """Reaching FINISH without a termination reason is a contract violation."""
    ctx = make_context()
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="Finish node requires a termination reason"):
        finish_node(state, FakeRuntime(ctx))


def test_finish_creates_agent_result(trace, make_context, make_state):
    """FINISH builds a stable AgentResult with reason, answer, and trace path."""
    ctx = make_context()
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.DONE
    state["answer"] = "final"

    result = finish_node(state, FakeRuntime(ctx))

    agent_result = result["result"]
    assert isinstance(agent_result, AgentResult)
    assert agent_result.reason == TerminationReason.DONE
    assert agent_result.answer == "final"
    assert agent_result.trace == str(ctx.trace.path)


def test_finish_result_numeric_fields(trace, make_context, make_state):
    """FINISH accurately populates numeric fields in AgentResult."""
    ctx = make_context()
    state = make_state(ctx)
    state["budget"].steps = 3
    state["budget"].tokens = 42
    state["budget"].cost_usd = 0.0123456
    state["history"] = [{"role": "user", "content": "a"}] * 5
    state["termination_reason"] = TerminationReason.DONE

    r = finish_node(state, FakeRuntime(ctx))["result"]

    assert r.steps == 3
    assert r.tokens == 42
    assert r.cost_usd == pytest.approx(0.012346)
    assert r.history_len == 5


def test_finish_missing_answer_returns_none(trace, make_context, make_state):
    """FINISH handles a missing answer gracefully.

    The trace record is read via ctx.trace.records() so the assertion does not
    depend on the JSONL separators or field ordering, and does not rely on the
    implicit trace fixture -> make_context wiring.
    """
    ctx = make_context()
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.BUDGET_TIME

    result = finish_node(state, FakeRuntime(ctx))

    assert result["result"].answer is None
    records = ctx.trace.records()
    assert records[-1]["t"] == "finish"
    assert records[-1]["answer_chars"] == 0


def test_finish_trace_finish_recorded(trace, make_context, make_state):
    """FINISH writes the final record to the trace file."""
    ctx = make_context()
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.DONE_UNVERIFIED
    state["answer"] = "x"

    finish_node(state, FakeRuntime(ctx))

    finish = [r for r in ctx.trace.records() if r["t"] == "finish"]
    assert len(finish) == 1
    assert finish[0]["reason"] == "done_unverified"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", [
    ({"termination_reason": TerminationReason.DONE}, "finish"),
    ({"pending_tool_calls": [{"name": "x"}]}, "execute"),
    ({"pending_answer": "a"}, "verify"),
    ({}, "decide"),
], ids=["done_to_finish", "tool_calls_to_execute", "answer_to_verify", "default_to_decide"])
def test_route_after_decide(state, expected):
    assert route_after_decide(state) == expected


def test_route_after_execute():
    assert route_after_execute({"termination_reason": TerminationReason.DONE}) == "finish"
    assert route_after_execute({}) == "decide"


def test_route_after_verify():
    assert route_after_verify({"termination_reason": TerminationReason.DONE}) == "finish"
    assert route_after_verify({}) == "decide"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_create_initial_state_copies_history():
    """The state history must be a copy, not an alias of the task history."""
    task = Task(
        system_prompt="s",
        initial_messages=[{"role": "user", "content": "a"}],
        budget_limits=BudgetLimits(
            max_steps=16,
            max_seconds=300.0,
            max_tokens=200_000,
            max_cost_usd=0.50,
        ),
    )
    state = create_initial_state(task)

    assert state["history"] == task.initial_messages
    assert state["history"] is not task.initial_messages
    assert state["budget"].limits.max_steps == task.budget_limits.max_steps
