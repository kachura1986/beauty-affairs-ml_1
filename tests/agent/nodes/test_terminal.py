import pytest
from dataclasses import replace

from services.agent.agent_graph.nodes import finish_node, verify_node
from services.agent.agent_graph.routing import route_after_decide, route_after_execute, route_after_verify
from services.agent.agent_graph.state import create_initial_state
from services.agent.domain import AgentResult, BudgetLimits, Finalization, Task, TerminationReason
from services.agent.trace import Trace

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
    """A failing postcondition rejects the answer and feeds the note back to DECIDE as an observation."""
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
    assert agent_result.finalization == Finalization.NONE


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
    """FINISH handles a missing answer gracefully (answer=None, answer_chars=0)."""
    ctx = make_context()
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.BUDGET_TIME

    result = finish_node(state, FakeRuntime(ctx))

    assert result["result"].answer is None
    assert result["result"].finalization == Finalization.NONE
    records = ctx.trace.records()
    assert records[-1]["t"] == "finish"
    assert records[-1]["answer_chars"] == 0
    assert records[-1]["finalization"] == "none"


def test_finish_uses_answer_from_budget_branch(trace, make_context, make_state):
    """A budget termination reason is carried through to AgentResult — not silently rewritten into DONE/DONE_UNVERIFIED."""
    ctx = make_context()
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.BUDGET_TOKENS
    state["answer"] = "kept"

    result = finish_node(state, FakeRuntime(ctx))

    assert result["result"].reason == TerminationReason.BUDGET_TOKENS
    assert result["result"].answer == "kept"
    records = ctx.trace.records()
    assert records[-1]["t"] == "finish"
    assert records[-1]["answer_chars"] == len("kept")


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
    assert finish[0]["finalization"] == "none"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", [
    ({"termination_reason": TerminationReason.DONE}, "finish"),
    ({"needs_finalization": True}, "finalize"),
    ({"needs_finalization": True, "termination_reason": TerminationReason.BUDGET_STEPS}, "finalize"),
    ({"pending_tool_calls": [{"name": "x"}]}, "execute"),
    ({"pending_answer": "a"}, "verify"),
    ({}, "decide"),
], ids=[
    "done_to_finish",
    "needs_finalization_to_finalize",
    "needs_finalization_overrides_termination",
    "tool_calls_to_execute",
    "answer_to_verify",
    "default_to_decide",
])
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


# ---------------------------------------------------------------------------
# AgentResult: finalization enum contract
# ---------------------------------------------------------------------------

def test_agent_result_finalization_enum():
    """Finalization.LLM/FALLBACK/NONE serialize to 'llm'/'fallback'/'none' (StrEnum)."""
    for value, expected in [
        (Finalization.LLM, "llm"),
        (Finalization.FALLBACK, "fallback"),
        (Finalization.NONE, "none"),
    ]:
        r = AgentResult(
            run_id="r1",
            reason=TerminationReason.BUDGET_STEPS,
            answer="x",
            steps=1,
            tokens=1,
            cost_usd=0.0,
            seconds=0.1,
            history_len=1,
            trace="/tmp/t",
            finalization=value,
        )
        assert r.model_dump(mode="json")["finalization"] == expected

    # Default is NONE.
    r = AgentResult(
        run_id="r1",
        reason=TerminationReason.DONE,
        answer="x",
        steps=1,
        tokens=1,
        cost_usd=0.0,
        seconds=0.1,
        history_len=1,
        trace="/tmp/t",
    )
    assert r.finalization == Finalization.NONE


# ---------------------------------------------------------------------------
# agent_name: per-agent attribution end-to-end (finish_node → AgentResult)
# ---------------------------------------------------------------------------

def test_finish_node_copies_trace_agent_name(trace, make_context, make_state, tmp_path):
    """finish_node copies trace.agent_name into AgentResult.agent_name.

    The shared `trace` fixture creates an unnamed Trace; a dedicated named
    trace is injected via dataclasses.replace so the test does not depend on
    fixture-internal state.
    """
    named_trace = Trace(tmp_path / 'named', agent_name='classifier_agent')
    try:
        ctx = replace(make_context(), trace=named_trace)
        state = make_state(ctx)
        state["termination_reason"] = TerminationReason.DONE
        state["answer"] = "final"

        result = finish_node(state, FakeRuntime(ctx))

        assert result["result"].agent_name == "classifier_agent"
    finally:
        named_trace.close()


def test_finish_node_agent_name_none_when_trace_has_no_name(trace, make_context, make_state):
    """An unnamed Trace (ad-hoc / test fixture) produces AgentResult.agent_name=None."""
    ctx = make_context()
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.DONE

    result = finish_node(state, FakeRuntime(ctx))

    assert result["result"].agent_name is None


def test_agent_result_agent_name_defaults_to_none():
    """Backward compatible: AgentResult constructed without agent_name keeps None."""
    r = AgentResult(
        run_id="r1",
        reason=TerminationReason.DONE_UNVERIFIED,
        answer="x",
        steps=1,
        tokens=1,
        cost_usd=0.0,
        seconds=0.1,
        history_len=1,
        trace="/tmp/t",
    )
    assert r.agent_name is None


def test_agent_result_agent_name_preserved():
    """agent_name flows through AgentResult and model_dump() unchanged."""
    r = AgentResult(
        run_id="r1",
        agent_name="classifier_agent",
        reason=TerminationReason.DONE_UNVERIFIED,
        answer="x",
        steps=1,
        tokens=1,
        cost_usd=0.0,
        seconds=0.1,
        history_len=1,
        trace="/tmp/t",
    )
    assert r.agent_name == "classifier_agent"
    assert r.model_dump()["agent_name"] == "classifier_agent"
