"""FINALIZE stage tests: finalize_node, deterministic fallback, end-to-end flow.

Covers unit behavior of finalize_node and its helpers, plus end-to-end
scenarios through the compiled graph for the "max_steps exhausted with
tool results in history" case. Existing per-stage tests live in
test_decide.py, test_execute.py, test_terminal.py.
"""
import logging
import time

import pytest

from services.agent.agent_graph.nodes import finalize_node
from services.agent.agent_graph._fallback import _deterministic_fallback, _has_tool_results
from services.agent.agent_graph.runner import run_graph
from services.agent.domain import (
    AgentPolicy,
    Budget,
    BudgetLimits,
    Finalization,
    Task,
    TerminationReason,
)
from services.agent.tools import Tool

from _testing import EchoArgs, FakeLLM, FakeRuntime, failing_tool, reply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finalize_state(ctx, *, history, last_tool_ok=None, steps=1):
    """
    Build a pre-finalize state.

    ``state['termination_reason']`` is set to BUDGET_STEPS and
    ``needs_finalization`` to True — the exact postcondition that
    ``decide_node`` establishes before routing to ``finalize_node``.
    """
    from services.agent.agent_graph.state import create_initial_state

    state = create_initial_state(ctx.task)
    state["budget"].steps = steps
    state["history"] = history
    state["last_tool_ok"] = last_tool_ok
    state["termination_reason"] = TerminationReason.BUDGET_STEPS
    state["needs_finalization"] = True
    return state


def _task(*, max_steps=1, max_seconds=300.0, max_tokens=200_000, max_cost_usd=0.50):
    return Task(
        system_prompt="sys",
        initial_messages=[{"role": "user", "content": "hi"}],
        budget_limits=BudgetLimits(
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
        ),
    )


# ---------------------------------------------------------------------------
# finalize_node — unit tests
# ---------------------------------------------------------------------------

async def test_finalize_llm_success(trace, make_context):
    """LLM succeeds: answer set, finalization=LLM, tools disabled, anchor present."""
    llm = FakeLLM(reply(text="final answer"))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["answer"] == "final answer"
    assert result["finalization"] == Finalization.LLM
    assert result["needs_finalization"] is False
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result
    assert llm.calls[0]["tools"] == []
    assert "You have reached the final step" in llm.calls[0]["system_prompt"]


async def test_finalize_skips_llm_when_time_exhausted(trace, make_context):
    """remaining_seconds() <= 0 → no LLM call, fallback, trace.error(budget_exhausted)."""
    llm = FakeLLM()
    ctx = make_context(llm=llm, max_steps=1, max_seconds=0.001)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )
    state["budget"].started = time.monotonic() - 10  # exhaust wall-clock

    result = await finalize_node(state, FakeRuntime(ctx))

    assert llm.calls == []
    assert result["answer"] is not None and "ok" in result["answer"]
    assert result["finalization"] == Finalization.FALLBACK
    assert result["needs_finalization"] is False
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result

    errors = [r for r in trace.records() if r.get("kind") == "finalization_fallback"]
    assert len(errors) == 1
    assert errors[0]["message"] == "budget_exhausted"


async def test_finalize_skips_llm_when_tokens_below_min(trace, make_context):
    """remaining_tokens() < MIN_EXPECTED_TOKENS → no LLM call, fallback."""
    from core.config import settings

    llm = FakeLLM()
    ctx = make_context(llm=llm, max_steps=1, max_tokens=settings.MIN_EXPECTED_TOKENS - 1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert llm.calls == []
    assert result["finalization"] == Finalization.FALLBACK
    assert result["answer"] is not None
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result
    assert any(
        r.get("kind") == "finalization_fallback" and r.get("message") == "budget_exhausted"
        for r in trace.records()
    )


async def test_finalize_does_not_block_on_cost_exhausted(trace, make_context):
    """Cost exhaustion is NOT a finalization pre-flight.

    Design decision: remaining_cost_usd() == 0 means the *accumulated* spend
    reached the limit, not that the next call is impossible. A free/local
    model can still answer at zero cost. The call proceeds, and an overrun
    (if any) is recorded as trace.error('finalization_overrun'), not as a
    pre-flight rejection.

    This test pins that decision so a future "symmetric" cost check does
    not silently break free-model finalization.
    """
    llm = FakeLLM(reply(text="final", in_tokens=0, out_tokens=0, cost_usd=0.0))
    ctx = make_context(llm=llm, max_steps=1, max_cost_usd=0.0001)
    state = _finalize_state(
        ctx, history=[{"role": "tool", "content": "ok"}], last_tool_ok="ok",
    )
    # Simulate that the accumulated cost already reached the limit before
    # entering finalize_node.
    state["budget"].cost_usd = 0.0001

    result = await finalize_node(state, FakeRuntime(ctx))

    # LLM was called despite cost exhaustion.
    assert len(llm.calls) == 1
    assert result["answer"] == "final"
    assert result["finalization"] == Finalization.LLM

    # And no pre-flight rejection was recorded.
    assert not any(
        r.get("kind") == "finalization_fallback" and r.get("message") == "budget_exhausted"
        for r in trace.records()
    )


async def test_finalize_llm_returns_tool_calls(trace, make_context):
    """tool_calls in the finalize reply are ignored; fallback used; no pending_tool_calls."""
    call = {"id": "c1", "name": "echo", "arguments": {}}
    llm = FakeLLM(reply(text="something", tool_calls=[call]))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["finalization"] == Finalization.FALLBACK
    assert result["answer"] is not None and "ok" in result["answer"]
    assert "pending_tool_calls" not in result
    assert result["needs_finalization"] is False
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result
    assert any(
        r.get("kind") == "finalization_fallback" and r.get("message") == "tool_calls_in_reply"
        for r in trace.records()
    )


async def test_finalize_llm_returns_empty_text(trace, make_context):
    """Empty reply text → fallback with trace.error(empty_reply)."""
    llm = FakeLLM(reply(text="   "))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["finalization"] == Finalization.FALLBACK
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result
    assert any(
        r.get("kind") == "finalization_fallback" and r.get("message") == "empty_reply"
        for r in trace.records()
    )


async def test_finalize_fallback_does_not_record_llm_usage(trace, make_context):
    """Fallback paths (budget exhausted, empty reply, tool_calls) must not
    call budget.record_llm — no LLM usage was consumed on those paths.

    Only the success path records tokens and cost (see
    test_finalize_records_llm_usage). A fallback that increments tokens
    would corrupt the finish.budget snapshot and break overrun detection.
    """
    from core.config import settings

    # Pre-flight rejection: tokens below MIN_EXPECTED_TOKENS.
    llm = FakeLLM()
    ctx = make_context(
        llm=llm, max_steps=1, max_tokens=settings.MIN_EXPECTED_TOKENS - 1,
    )
    state = _finalize_state(
        ctx, history=[{"role": "tool", "content": "ok"}], last_tool_ok="ok",
    )
    tokens_before = state["budget"].tokens
    cost_before = state["budget"].cost_usd

    await finalize_node(state, FakeRuntime(ctx))

    assert state["budget"].tokens == tokens_before
    assert state["budget"].cost_usd == cost_before
    assert llm.calls == []


async def test_finalize_rejects_non_budget_steps(trace, make_context):
    """A non-BUDGET_STEPS termination reason is a contract violation."""
    ctx = make_context()
    state = _finalize_state(ctx, history=[{"role": "tool", "content": "ok"}])
    state["termination_reason"] = TerminationReason.BUDGET_TIME

    with pytest.raises(RuntimeError, match="BUDGET_STEPS"):
        await finalize_node(state, FakeRuntime(ctx))


async def test_finalize_defensive_no_needs_finalization(trace, make_context):
    """Defensive: needs_finalization=False → NONE without an LLM call."""
    llm = FakeLLM()
    ctx = make_context(llm=llm)
    state = _finalize_state(ctx, history=[{"role": "tool", "content": "ok"}])
    state["needs_finalization"] = False

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["finalization"] == Finalization.NONE
    assert result["needs_finalization"] is False
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result
    assert llm.calls == []


async def test_finalize_all_error_history_returns_fallback_explanation(trace, make_context):
    """All-error history: deterministic fallback explains the failure (not None)."""
    llm = FakeLLM(reply(text=""))  # empty LLM reply → fallback branch
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ERROR execution_failed. boom."}],
        last_tool_ok=None,
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["answer"] is not None
    assert "ERROR" in result["answer"]
    assert result["finalization"] == Finalization.FALLBACK
    assert result["needs_finalization"] is False
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result


async def test_finalize_no_tool_messages_returns_none(trace, make_context):
    """No tool messages at all → fallback returns None → NONE.

    Also pins that the NONE path does not leak termination_reason back into
    the node result — the contract is identical on every return path of
    finalize_node, including the one that leaves answer=None.
    """
    llm = FakeLLM(reply(text=""))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(ctx, history=[], last_tool_ok=None)

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["answer"] is None
    assert result["finalization"] == Finalization.NONE
    assert result["needs_finalization"] is False
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result


async def test_finalize_logs_internal_exception(trace, make_context, caplog):
    """LLM exception → logged via _log_internal_exception + trace.error(llm_error)."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    with caplog.at_level(logging.ERROR):
        result = await finalize_node(state, FakeRuntime(ctx))

    assert any("Finalization LLM call failed" in r.message for r in caplog.records)
    assert result["finalization"] == Finalization.FALLBACK
    assert result["answer"] is not None and "ok" in result["answer"]
    # finalize_node must not overwrite termination_reason set by decide_node.
    assert "termination_reason" not in result
    assert any(
        r.get("kind") == "finalization_fallback" and r.get("message") == "llm_error"
        for r in trace.records()
    )


async def test_finalize_records_llm_timeout_when_budget_exhausted(trace, make_context, monkeypatch):
    """LLM exception with exhausted wall-clock budget → message='llm_timeout', not 'llm_error'.

    Mirrors `test_request_llm_timeout_returns_budget_time`: the distinction is
    'budget exhausted after the exception', not 'TimeoutError'. The pinned call
    count guards against extra budget checks.
    """
    llm = FakeLLM(errors=[TimeoutError("slow")])
    ctx = make_context(llm=llm, max_steps=1, max_seconds=0.001)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    elapsed_calls: list[int] = []
    times = iter([0.0, 0.0, 2.0, 2.0])

    def fake_elapsed(self):
        elapsed_calls.append(1)
        return next(times)

    monkeypatch.setattr(Budget, "elapsed", fake_elapsed)

    await finalize_node(state, FakeRuntime(ctx))

    errors = [r for r in trace.records() if r.get("kind") == "finalization_fallback"]
    assert len(errors) == 1
    assert errors[0]["message"] == "llm_timeout"
    assert len(elapsed_calls) == 4


async def test_finalize_records_overrun_in_trace(trace, make_context):
    """Successful finalize that pushes over hard budget records finalization_overrun.

    Termination reason is preserved (BUDGET_STEPS is set before finalize_node
    runs); the overrun is recorded as a diagnostic trace error.

    ``max_tokens`` is set just above ``settings.MIN_EXPECTED_TOKENS`` so the
    pre-flight passes (``remaining_tokens() >= MIN_EXPECTED_TOKENS``) and the
    LLM call actually happens; the oversized ``in_tokens`` on the reply then
    pushes the run over the token budget after ``record_llm``.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="final", in_tokens=10_000_000, out_tokens=0, cost_usd=0.0))
    ctx = make_context(
        llm=llm,
        max_steps=1,
        max_tokens=settings.MIN_EXPECTED_TOKENS + 100,
    )
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["finalization"] == Finalization.LLM
    assert result["answer"] == "final"
    overrun = [r for r in trace.records() if r.get("kind") == "finalization_overrun"]
    assert len(overrun) == 1
    assert overrun[0]["message"] == TerminationReason.BUDGET_TOKENS.value
    # The termination reason itself is unchanged.
    assert state["termination_reason"] == TerminationReason.BUDGET_STEPS


async def test_finalize_overrun_does_not_rewrite_termination_reason(trace, make_context):
    """Overrun on finalize keeps BUDGET_STEPS as the termination reason.

    The overrun is diagnostic only: trace.error('finalization_overrun') is
    written, but state['termination_reason'] must remain BUDGET_STEPS so
    finish_node reports the original reason and the AgentResult reflects
    that the run ended on step exhaustion, not on a token overrun during
    finalization.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="final", in_tokens=10_000_000, out_tokens=0, cost_usd=0.0))
    ctx = make_context(
        llm=llm, max_steps=1, max_tokens=settings.MIN_EXPECTED_TOKENS + 100,
    )
    state = _finalize_state(
        ctx, history=[{"role": "tool", "content": "ok"}], last_tool_ok="ok",
    )
    reason_before = state["termination_reason"]

    result = await finalize_node(state, FakeRuntime(ctx))

    # Overrun recorded, reason untouched, and the node's return dict does not
    # carry a termination_reason key (the service must not leak one).
    assert result["finalization"] == Finalization.LLM
    assert "termination_reason" not in result
    assert state["termination_reason"] == reason_before
    assert any(r.get("kind") == "finalization_overrun" for r in trace.records())


async def test_finalize_does_not_increment_steps(trace, make_context):
    """The finalization call must not consume a step (contract from BudgetLimits).

    The whole point of the extra call is to bypass ``can_start_step()`` for
    exactly one call while leaving ``budget.steps`` at ``max_steps``.
    """
    llm = FakeLLM(reply(text="final"))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )
    assert state["budget"].steps == 1

    await finalize_node(state, FakeRuntime(ctx))

    assert state["budget"].steps == 1  # not 2


async def test_finalize_records_llm_usage(trace, make_context):
    """Tokens and cost of the finalization call are recorded on the budget.

    Without ``record_llm`` the overrun detection (Defect D) would be blind to
    token and cost overruns, and the trace's ``finish.budget`` snapshot would
    underreport the true cost of the run.
    """
    llm = FakeLLM(reply(text="final", in_tokens=5, out_tokens=3, cost_usd=0.0001))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "content": "ok"}],
        last_tool_ok="ok",
    )

    await finalize_node(state, FakeRuntime(ctx))

    assert state["budget"].tokens == 8
    assert state["budget"].cost_usd == pytest.approx(0.0001)


async def test_finalize_writes_finalize_record_on_success(trace, make_context):
    """A successful finalization writes a t='finalize' record with the call's usage.

    This separates the finalization cost/latency from decision steps in the
    trace (`t='step'` vs `t='finalize'`), so post-mortem can attribute usage
    without diffing `finish.budget`. No record is written on fallback paths —
    those are covered by `finalization_fallback` error records.
    """
    llm = FakeLLM(reply(
        text="final answer",
        in_tokens=10,
        out_tokens=5,
        cost_usd=0.001,
    ))
    ctx = make_context(llm=llm, max_steps=1)
    state = _finalize_state(
        ctx,
        history=[{"role": "tool", "name": "t", "tool_call_id": "c1", "content": "ok"}],
        last_tool_ok="ok",
    )

    result = await finalize_node(state, FakeRuntime(ctx))

    assert result["finalization"] == Finalization.LLM
    assert result["answer"] == "final answer"

    finalize_records = [r for r in trace.records() if r.get("t") == "finalize"]
    assert len(finalize_records) == 1

    record = finalize_records[0]
    assert record["model"] == "fake"
    assert record["in_tokens"] == 10
    assert record["out_tokens"] == 5
    assert record["decision"] == "answer"
    assert record["tools"] == []
    assert record["history_msgs"] == 1
    assert "system_hash" in record
    assert "budget" in record


async def test_finalize_node_resets_needs_finalization(trace, make_context):
    """All exit paths reset needs_finalization — the flag never leaks to finish."""
    # Success path
    llm1 = FakeLLM(reply(text="final"))
    ctx1 = make_context(llm=llm1, max_steps=1)
    state1 = _finalize_state(ctx1, history=[{"role": "tool", "content": "ok"}], last_tool_ok="ok")
    r1 = await finalize_node(state1, FakeRuntime(ctx1))
    assert r1["needs_finalization"] is False

    # Empty-reply / fallback path
    llm2 = FakeLLM(reply(text=""))
    ctx2 = make_context(llm=llm2, max_steps=1)
    state2 = _finalize_state(ctx2, history=[{"role": "tool", "content": "ok"}], last_tool_ok="ok")
    r2 = await finalize_node(state2, FakeRuntime(ctx2))
    assert r2["needs_finalization"] is False

    # Defensive branch
    llm3 = FakeLLM()
    ctx3 = make_context(llm=llm3)
    state3 = _finalize_state(ctx3, history=[])
    state3["needs_finalization"] = False
    r3 = await finalize_node(state3, FakeRuntime(ctx3))
    assert r3["needs_finalization"] is False


# ---------------------------------------------------------------------------
# Deterministic fallback / _has_tool_results — direct unit tests
# ---------------------------------------------------------------------------

def test_deterministic_fallback_uses_last_tool_ok():
    """When last_tool_ok is set, it is used verbatim regardless of history."""
    history = [
        {"role": "tool", "content": "ERROR execution_failed. boom."},
        {"role": "tool", "content": "older-ok"},
    ]
    result = _deterministic_fallback(history, '{"id": 1, "name": "ok"}')

    assert result is not None
    assert result.startswith(
        "The model could not produce a final answer within the step limit."
    )
    assert result.endswith('\n{"id": 1, "name": "ok"}')


def test_deterministic_fallback_returns_error_explanation_when_all_errors():
    """All-error history (last_tool_ok=None) yields an error explanation, not None."""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "ERROR execution_failed. boom."},
    ]
    result = _deterministic_fallback(history, None)

    assert result is not None
    assert "ERROR" in result


def test_deterministic_fallback_returns_none_when_no_tools():
    """No tool messages at all → None (caller leaves answer=None)."""
    history = [{"role": "user", "content": "hi"}]
    result = _deterministic_fallback(history, None)

    assert result is None


def test_deterministic_fallback_empty_string_last_tool_ok_takes_success_branch():
    """Empty-string last_tool_ok is treated as "set" (``is not None``).

    ``_deterministic_fallback`` checks ``last_tool_ok is not None``, so a
    corrupted or edge-case empty string — e.g. a tool that succeeded with
    empty content, or a bug that reset the field to "" — produces an answer
    embedding it verbatim instead of falling through to the error-explanation
    path.

    This test pins the current boundary. If the implementation is tightened
    to ``if last_tool_ok:`` (arguably a better fallback), this test must be
    updated deliberately — the empty string would then take the
    error-explanation path.
    """
    history = [{"role": "tool", "content": "ERROR execution_failed. boom."}]

    result = _deterministic_fallback(history, "")

    assert result is not None
    assert "Here is the most recent tool result:" in result
    # The error-explanation path would include the error content; the success
    # branch must not.
    assert "ERROR" not in result


def test_has_tool_results_counts_synthetic_errors():
    """Synthetic error tool messages count as tool results."""
    history = [
        {
            "role": "tool",
            "content": "ERROR execution_failed. Tool 'x' was not executed: run is terminating (repeated_action).",
        },
    ]
    assert _has_tool_results(history) is True


def test_has_tool_results_false_for_empty_history():
    assert _has_tool_results([{"role": "user", "content": "hi"}]) is False


# ---------------------------------------------------------------------------
# End-to-end graph scenarios
# ---------------------------------------------------------------------------

async def test_max_steps_1_tool_call_finalizes(trace, make_registry):
    """max_steps=1 + tool_call → execute → decide(BUDGET_STEPS) → finalize → LLM answer."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    llm = FakeLLM(
        reply(text="", tool_calls=[call]),
        reply(text="Final synthesized answer"),
    )
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS
    assert result.answer == "Final synthesized answer"
    assert result.finalization == Finalization.LLM


async def test_max_steps_1_tool_call_fails_finalizes(trace, make_registry):
    """max_steps=1 + failing tool → finalize runs; LLM synthesizes the error explanation."""
    call = {"id": "c1", "name": "failing", "arguments": {}}
    llm = FakeLLM(
        reply(text="", tool_calls=[call]),
        reply(text="The tool failed; here is what we know."),
    )
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(failing_tool()),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS
    assert result.answer is not None
    assert result.finalization == Finalization.LLM


async def test_max_steps_1_tool_fails_and_llm_fails_returns_fallback_explanation(trace, make_registry):
    """Tool fails + finalize LLM returns empty → deterministic fallback explains the error."""
    call = {"id": "c1", "name": "failing", "arguments": {}}
    llm = FakeLLM(
        reply(text="", tool_calls=[call]),
        reply(text=""),  # empty finalize reply
    )
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(failing_tool()),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS
    assert result.answer is not None
    assert "ERROR" in result.answer
    assert result.finalization == Finalization.FALLBACK


async def test_multiple_tool_calls_at_max_steps(trace, make_registry):
    """Multiple tool_calls at max_steps: all executed, finalize synthesizes the answer."""
    calls = [
        {"id": "c1", "name": "echo", "arguments": {"value": "a"}},
        {"id": "c2", "name": "echo", "arguments": {"value": "b"}},
    ]
    llm = FakeLLM(
        reply(text="", tool_calls=calls),
        reply(text="Here are both results."),
    )
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS
    assert result.answer is not None
    assert result.finalization == Finalization.LLM
    # Both tool messages were paired into history (initial user + assistant + 2 tool + finalize)
    assert result.history_len >= 4


async def test_normal_path_no_finalize(trace, make_registry):
    """Normal completion does not trigger finalization."""
    llm = FakeLLM(reply(text="direct answer"))
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    result = await run_graph(
        task=_task(max_steps=5),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.DONE_UNVERIFIED
    assert result.answer == "direct answer"
    assert result.finalization == Finalization.NONE


async def test_finalize_on_max_steps_false_legacy(trace, make_registry):
    """finalize_on_max_steps=False preserves the legacy answer=None behavior."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    policy = AgentPolicy(empty_limit=3, repeat_limit=3, finalize_on_max_steps=False)

    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS
    assert result.answer is None
    assert result.finalization == Finalization.NONE


async def test_finalize_not_called_without_tool_results(trace, make_registry):
    """Empty reply + no tool results → needs_finalization not set, finalize not called."""
    llm = FakeLLM(reply(text=""))  # empty on step 1
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS
    assert result.answer is None
    assert result.finalization == Finalization.NONE


async def test_max_steps_1_does_not_hit_recursion_limit(trace, make_registry):
    """Explicit regression guard: max_steps=1 + tool_call does not trip recursion_limit."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    llm = FakeLLM(
        reply(text="", tool_calls=[call]),
        reply(text="answer"),
    )
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    # GraphRecursionError must NOT escape: max_steps=1 + finalize fits
    # well within the configured recursion_limit (max(64, 1*4+16) = 64).
    result = await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_STEPS


async def test_budget_time_with_tool_results_does_not_finalize(trace, make_registry):
    """Finalization is gated to BUDGET_STEPS; BUDGET_TIME never triggers it.

    When the wall-clock budget is exhausted mid-execute, the termination
    reason is BUDGET_TIME, not BUDGET_STEPS, so ``decide_node`` is never
    reached again and finalization is not attempted — even though tool
    messages are already in history. This pins the decision to gate
    finalization to step-exhaustion only and prevents a future refactor
    from silently widening it to time/token/cost exhaustion, where there
    would be no budget left for the extra call anyway.
    """
    import asyncio

    async def slow_fn(**kwargs):
        await asyncio.Event().wait()
        return "unreachable"  # pragma: no cover

    call = {"id": "c1", "name": "slow", "arguments": {}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)
    registry = make_registry(
        Tool(name="slow", description="", fn=slow_fn, args_model=EchoArgs)
    )

    result = await run_graph(
        task=_task(max_steps=5, max_seconds=0.05),
        tools=registry,
        llm=llm,
        trace=trace,
        policy=policy,
    )

    assert result.reason == TerminationReason.BUDGET_TIME
    assert result.answer is None
    assert result.finalization == Finalization.NONE


async def test_finalization_in_trace_finish(trace, make_registry):
    """The finish trace record carries finalization='llm' and answer_chars > 0.

    Also asserts that step-record fields do not leak into the finish record:
    those fields belong to ``t='step'`` and their presence in ``t='finish'``
    would indicate the two record shapes were accidentally merged in
    ``Trace.finish`` (regression guard for a "copied step() and forgot to
    trim window_*" refactor).
    """
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    llm = FakeLLM(
        reply(text="", tool_calls=[call]),
        reply(text="Final synthesized answer"),
    )
    policy = AgentPolicy(empty_limit=3, repeat_limit=3)

    await run_graph(
        task=_task(max_steps=1),
        tools=make_registry(),
        llm=llm,
        trace=trace,
        policy=policy,
    )

    finish = [r for r in trace.records() if r["t"] == "finish"]
    assert len(finish) == 1
    assert finish[0]["finalization"] == "llm"
    assert finish[0]["answer_chars"] > 0

    for step_field in (
            "window_hash", "window_msgs", "system_hash", "system_chars",
            "model", "in_tokens", "out_tokens", "latency_ms", "decision", "tools",
    ):
        assert step_field not in finish[0], (
            f"{step_field!r} leaked from a step record into the finish record"
        )
