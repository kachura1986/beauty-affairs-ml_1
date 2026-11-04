"""DECIDE stage tests: _check_budget, _request_llm, decide_node.

Grouped by graph stage: this file maps 1:1 to the DECIDE node and the
helpers it relies on. EXECUTE lives in test_execute.py, terminal stages
(VERIFY/FINISH/routing/state) in test_terminal.py.
"""
import time

import pytest

from services.agent.agent_graph.nodes import (
    _check_budget,
    _request_llm,
    decide_node,
)
from services.agent.context import AgentContextWindow
from services.agent.domain import (
    AgentPolicy,
    Budget,
    BudgetLimits,
    TerminationReason,
)

from _testing import FakeLLM, FakeRuntime, make_context_window, reply


# ---------------------------------------------------------------------------
# _check_budget
# ---------------------------------------------------------------------------

def test_check_budget_steps_exhausted():
    """An exhausted step budget yields BUDGET_STEPS."""
    budget = Budget(limits=BudgetLimits(max_steps=1, max_seconds=10, max_tokens=10, max_cost_usd=1))
    budget.steps = 1
    assert _check_budget(budget) == TerminationReason.BUDGET_STEPS


def test_check_budget_time_exceeded():
    """An expired wall-clock budget yields BUDGET_TIME even when steps remain.

    ``BudgetLimits`` enforces ``max_seconds > 0``, so the budget is expired
    by shifting its start time into the past instead of setting the limit to zero.
    """
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=0.001, max_tokens=10, max_cost_usd=1))
    budget.started = time.monotonic() - 10
    assert _check_budget(budget) == TerminationReason.BUDGET_TIME


def test_check_budget_ok():
    """A fresh budget with all limits intact yields None."""
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=10, max_cost_usd=1))
    assert _check_budget(budget) is None


def test_check_budget_steps_take_priority_over_time():
    """Steps exhaustion is checked before time exhaustion."""
    budget = Budget(limits=BudgetLimits(max_steps=1, max_seconds=0.001, max_tokens=10, max_cost_usd=1))
    budget.steps = 1
    budget.started = time.monotonic() - 10
    assert _check_budget(budget) == TerminationReason.BUDGET_STEPS


def test_exceeded_without_steps_time_before_tokens():
    """Time exhaustion is checked before token exhaustion."""
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=0.001, max_tokens=1, max_cost_usd=1))
    budget.tokens = 100
    budget.started = time.monotonic() - 10
    assert budget.exceeded_without_steps() == TerminationReason.BUDGET_TIME


# ---------------------------------------------------------------------------
# _request_llm
# ---------------------------------------------------------------------------

async def test_request_llm_success(trace, make_context):
    """A successful LLM call increments steps and records tokens and cost."""
    llm = FakeLLM(reply(text="hello", in_tokens=5, out_tokens=3, cost_usd=0.001))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    cw = make_context_window(
        system="custom-system-prompt",
        messages=[{"role": "user", "content": "hi"}],
    )
    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=cw,
    )

    assert reason is None
    assert result is not None and result.text == "hello"
    assert budget.steps == 1
    assert budget.tokens == 8
    assert budget.cost_usd == pytest.approx(0.001)
    assert llm.calls[0]["tools"] == ctx.tools.schemas()
    assert llm.calls[0]["timeout"] == pytest.approx(10.0, abs=0.1)
    assert llm.calls[0]["messages"] is cw.messages
    assert llm.calls[0]["system_prompt"] == "custom-system-prompt"
    assert llm.calls[0]["system_prompt"] == cw.system


async def test_request_llm_chat_and_trace_get_same_messages_object(trace, make_context, monkeypatch):
    """Invariant: chat() and trace.step() receive the same messages object.

    Regression: if someone copies or slices the list between the calls,
    the window_hash in the trace no longer matches what was sent to the model.
    """
    import services.agent.agent_graph.nodes as nodes_mod

    seen = {}
    real_step = nodes_mod.Trace.step

    def spy_step(self, n, window, reply, budget, latency_ms, *, system_prompt):
        seen["window"] = window
        return real_step(self, n, window, reply, budget, latency_ms, system_prompt=system_prompt)

    monkeypatch.setattr(nodes_mod.Trace, "step", spy_step)

    messages = [{"role": "user", "content": "hi"}]
    cw = make_context_window(system="custom-system", messages=messages)
    llm = FakeLLM(reply(text="ok"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, context_window=cw)

    assert llm.calls[0]["messages"] is cw.messages
    assert seen["window"] is cw.messages
    assert seen["window"] is llm.calls[0]["messages"]


async def test_request_llm_timeout_returns_budget_time(trace, make_context, monkeypatch):
    """An LLM exception on an exhausted wall-clock budget yields BUDGET_TIME.

    ``_request_llm`` re-checks the remaining time after the exception, so a
    raised error must be classified as BUDGET_TIME rather than LLM_ERROR when
    the run has already run out of time during the call.
    """
    llm = FakeLLM(errors=[TimeoutError("slow")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=1.0, max_tokens=1000, max_cost_usd=1))

    # Emulate that the LLM call took longer than the remaining budget (1.0s).
    # Use a fallback to prevent StopIteration when budget.snapshot() calls elapsed() again.
    # Patch the method on the class: Budget uses validate_assignment=True, so an
    # instance-level setattr of a non-field attribute (such as a replacement
    # `elapsed`) is rejected by Pydantic. monkeypatch restores it after the test.
    times = iter([0.0, 2.0])
    monkeypatch.setattr(Budget, "elapsed", lambda self: next(times, 2.0))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TIME
    assert len(llm.calls) == 1


async def test_request_llm_llm_error(trace, make_context):
    """An LLM exception with remaining time yields LLM_ERROR."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR


async def test_request_llm_error_still_increments_steps(trace, make_context):
    """Steps are incremented even if the LLM call fails, as the step was started."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert budget.steps == 1


async def test_request_llm_without_chat_method_returns_llm_error(trace, make_context):
    """An LLM object lacking a chat method yields LLM_ERROR."""

    class NoChatLLM:
        model_name = "fake"

    ctx = make_context(llm=NoChatLLM())
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR


async def test_request_llm_exceeded_after_successful_call(trace, make_context):
    """A successful call that pushes usage over the limit yields BUDGET_TOKENS
    but preserves the reply — decide_node decides if it qualifies as a final answer.

    ``max_tokens == MIN_EXPECTED_TOKENS`` keeps the preflight check from firing
    (strict '<'), while ``in_tokens == MIN_EXPECTED_TOKENS + 1`` guarantees the
    post-call ``exceeded_without_steps()`` returns BUDGET_TOKENS. ``out_tokens``
    is pinned to 0 so the overrun comes from the input side, not from output
    noise that could mask a regression.
    """
    from core.config import settings

    llm = FakeLLM(reply(
        text="x",
        in_tokens=settings.MIN_EXPECTED_TOKENS + 1,
        out_tokens=0,
        cost_usd=0,
    ))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS,
        max_cost_usd=1,
    ))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert reason == TerminationReason.BUDGET_TOKENS
    assert result is not None
    assert result.text == "x"


async def test_request_llm_timeout_capped_by_remaining_budget(trace, make_context):
    """The LLM call timeout equals the remaining budget when it is below the cap.

    With the default LLM_DEFAULT_TIMEOUT_SECONDS (120s), a 10s budget must
    cap the call at ~10s rather than letting it run for the full default.
    """
    llm = FakeLLM(reply(text="hello"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert llm.calls[0]["timeout"] == pytest.approx(10.0, abs=0.1)


async def test_request_llm_timeout_capped_by_default(trace, make_context, monkeypatch):
    """The LLM call timeout equals the default cap when the budget is larger.

    A 300s budget must not force a call to wait the whole 300s when
    LLM_DEFAULT_TIMEOUT_SECONDS is 5s: min(300, 5) == 5.
    """
    from core.config import settings
    monkeypatch.setattr(settings, "LLM_DEFAULT_TIMEOUT_SECONDS", 5.0)

    llm = FakeLLM(reply(text="hello"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=300, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert llm.calls[0]["timeout"] == pytest.approx(5.0, abs=0.1)


async def test_request_llm_preflight_tokens_below_minimum(trace, make_context):
    """A token budget below MIN_EXPECTED_TOKENS short-circuits before chat().

    The check is strict '<', so a caller that lands exactly on the threshold
    must still be allowed to try (see the boundary test). A clearly
    insufficient budget must never reach the LLM.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="never"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS - 1,
        max_cost_usd=1,
    ))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TOKENS
    assert llm.calls == []


async def test_request_llm_preflight_tokens_at_boundary_allowed(trace, make_context):
    """Exactly MIN_EXPECTED_TOKENS remaining is still allowed to call the LLM.

    The pre-flight uses strict '<', so the boundary value must pass through
    to chat() rather than terminating early.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="ok"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS,
        max_cost_usd=1,
    ))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert reason is None
    assert result is not None
    assert len(llm.calls) == 1


async def test_request_llm_preflight_tokens_emits_trace(trace, make_context):
    """The pre-flight rejection is recorded in the trace with kind='preflight_tokens'.

    Parses JSONL via Trace.records() rather than substring-matching, so the
    check is independent of the trace's separators and field ordering.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="never"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS - 1,
        max_cost_usd=1,
    ))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert any(r.get("kind") == "preflight_tokens" for r in trace.records())


# ---------------------------------------------------------------------------
# decide_node
# ---------------------------------------------------------------------------

async def test_decide_returns_budget_steps_when_no_steps_left(trace, make_context, make_state):
    """An exhausted step budget short-circuits DECIDE before the LLM is called."""
    llm = FakeLLM()
    ctx = make_context(llm=llm, max_steps=1)
    state = make_state(ctx)
    state["budget"].steps = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_STEPS
    assert llm.calls == []


async def test_decide_raises_if_llm_returns_nothing(trace, make_context, make_state, monkeypatch):
    """A (None, None) result from _request_llm is a contract violation."""
    import services.agent.agent_graph.nodes as nodes_mod
    async def fake_request_llm(**kwargs):
        return None, None

    monkeypatch.setattr(nodes_mod, "_request_llm", fake_request_llm)

    ctx = make_context()
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="no reply"):
        await decide_node(state, FakeRuntime(ctx))


async def test_decide_empty_response_increments_and_adds_observation(trace, make_context, make_state):
    """An empty LLM reply increments the empty counter and appends an observation."""
    llm = FakeLLM(reply(text="   "))
    ctx = make_context(llm=llm, policy=AgentPolicy(empty_limit=2, repeat_limit=3))
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["consecutive_empty_steps"] == 1
    assert result["history"][0]["role"] == "user"
    assert "[observation]" in result["history"][0]["content"]
    assert "termination_reason" not in result
    assert result["budget"] is state["budget"]


async def test_decide_empty_response_exceeds_limit_returns_no_progress(trace, make_context, make_state):
    """Crossing the empty-response limit terminates with NO_PROGRESS."""
    llm = FakeLLM(reply(text=""))
    ctx = make_context(llm=llm, policy=AgentPolicy(empty_limit=1, repeat_limit=3))
    state = make_state(ctx)
    state["consecutive_empty_steps"] = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.NO_PROGRESS
    assert result["budget"] is state["budget"]


async def test_decide_text_answer_sets_pending_answer(trace, make_context, make_state):
    """A text-only reply becomes pending_answer and resets the empty counter."""
    llm = FakeLLM(reply(text="final answer"))
    ctx = make_context(llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_answer"] == "final answer"
    assert "pending_tool_calls" not in result
    assert result["consecutive_empty_steps"] == 0
    assert "termination_reason" not in result


async def test_decide_tool_calls_sets_pending_tool_calls_and_history(trace, make_context, make_state):
    """A reply with tool_calls becomes pending and is echoed into history."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    ctx = make_context(llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_tool_calls"] == [call]
    assert result["pending_answer"] is None
    assert result["history"][0]["role"] == "assistant"
    assert result["history"][0]["tool_calls"] == [call]
    assert "termination_reason" not in result


async def test_decide_tool_calls_clears_stale_pending_answer(trace, make_context, make_state):
    """A tool-call reply overwrites a stale pending_answer from a previous step."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    ctx = make_context(llm=llm)
    state = make_state(ctx)
    state["pending_answer"] = "stale"

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_answer"] is None


# ---------------------------------------------------------------------------
# decide_node: budget reason with a preserved reply
#
# The budget branch of `decide_node` qualifies a reply as a final answer only
# when it has no tool_calls and non-blank text. These tests pin each condition
# in isolation; the shared setup mirrors the exact preflight boundary used by
# `test_request_llm_exceeded_after_successful_call` — `max_tokens` equals
# MIN_EXPECTED_TOKENS so the preflight does not fire, while `in_tokens` is one
# over so the post-call `exceeded_without_steps()` returns BUDGET_TOKENS.
# ---------------------------------------------------------------------------

async def test_decide_preserves_answer_when_budget_exceeded_after_final_text(
        trace, make_context, make_state,
):
    """Scenario 6/7: a final-text reply that pushes usage over the token
    budget must become `answer`, not be dropped into an empty AgentResult."""
    from core.config import settings

    llm = FakeLLM(reply(
        text="final answer",
        in_tokens=settings.MIN_EXPECTED_TOKENS + 1,
        out_tokens=0,
        cost_usd=0,
    ))
    ctx = make_context(llm=llm, max_tokens=settings.MIN_EXPECTED_TOKENS)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TOKENS
    assert result["answer"] == "final answer"
    assert result["budget"] is state["budget"]


async def test_decide_does_not_preserve_tool_call_as_answer(trace, make_context, make_state):
    """A tool_call reply at budget exhaustion is not a final answer —
    the model asked to continue, not to answer."""
    from core.config import settings

    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(
        text="",
        tool_calls=[call],
        in_tokens=settings.MIN_EXPECTED_TOKENS + 1,
        out_tokens=0,
        cost_usd=0,
    ))
    ctx = make_context(llm=llm, max_tokens=settings.MIN_EXPECTED_TOKENS)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TOKENS
    assert "answer" not in result


async def test_decide_does_not_preserve_empty_text_as_answer(trace, make_context, make_state):
    """A blank-text reply at budget exhaustion is not a final answer."""
    from core.config import settings

    llm = FakeLLM(reply(
        text="   ",
        in_tokens=settings.MIN_EXPECTED_TOKENS + 1,
        out_tokens=0,
        cost_usd=0,
    ))
    ctx = make_context(llm=llm, max_tokens=settings.MIN_EXPECTED_TOKENS)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TOKENS
    assert "answer" not in result


async def test_decide_tool_call_with_text_at_budget_not_preserved(trace, make_context, make_state):
    """Even if the reply carries both text and tool_calls, tool_calls make it
    non-final — `answer` must stay absent."""
    from core.config import settings

    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(
        text="intermediate thought",
        tool_calls=[call],
        in_tokens=settings.MIN_EXPECTED_TOKENS + 1,
        out_tokens=0,
        cost_usd=0,
    ))
    ctx = make_context(llm=llm, max_tokens=settings.MIN_EXPECTED_TOKENS)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TOKENS
    assert "answer" not in result


# ---------------------------------------------------------------------------
# decide_node: integration guards over the preflight and except branches
#
# Both branches of `_request_llm` return (None, reason). The `reply is not
# None` guard in the budget branch keeps `answer` absent — and must not raise
# on a None reply. These tests exercise that contract end-to-end through
# decide_node, not just at the _request_llm boundary.
# ---------------------------------------------------------------------------

async def test_decide_preflight_budget_does_not_set_answer(trace, make_context, make_state):
    """decide_node tolerates (None, reason) from the preflight path without
    setting `answer` and without raising."""
    from core.config import settings

    llm = FakeLLM(reply(text="never"))
    ctx = make_context(llm=llm, max_tokens=settings.MIN_EXPECTED_TOKENS - 1)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TOKENS
    assert "answer" not in result
    assert llm.calls == []


async def test_decide_llm_error_does_not_set_answer(trace, make_context, make_state):
    """A failed LLM call with reason != None does not qualify as an answer.

    The except branch of `_request_llm` returns (None, reason); the new
    contract ("reply may be non-None when reason != None") applies only to
    the success path, so this test pins the asymmetry.
    """
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.LLM_ERROR
    assert "answer" not in result


async def test_decide_emits_budget_warning_once(trace, make_context, make_state):
    """The soft budget warning sets the emitted flag exactly once and is passed to LLM."""
    llm = FakeLLM(reply(text="a"), reply(text="b"))
    ctx = make_context(llm=llm, max_steps=10)
    state = make_state(ctx)
    state["budget"].cost_usd = 0.45  # 90% of the default 0.50 cost budget

    first = await decide_node(state, FakeRuntime(ctx))
    assert first["budget_warning_emitted"] is True
    assert "WARNING: more than 80%" in llm.calls[0]["system_prompt"]

    state["budget_warning_emitted"] = True
    second = await decide_node(state, FakeRuntime(ctx))
    assert second["budget_warning_emitted"] is True
    assert "WARNING: more than 80%" not in llm.calls[1]["system_prompt"]


async def test_decide_budget_warning_not_reemitted(trace, make_context, make_state):
    """Once emitted, the warning flag stays True on subsequent steps."""
    llm = FakeLLM(reply(text="a"))
    ctx = make_context(llm=llm, max_steps=10)
    state = make_state(ctx)
    state["budget"].cost_usd = 0.45
    state["budget_warning_emitted"] = True

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["budget_warning_emitted"] is True
    assert "WARNING: more than 80%" not in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize("cost_usd,expected_emitted", [
    (0.0, False),
    (0.45, True),
])
async def test_decide_context_overflow_preserves_budget_warning_flag(
        trace, make_context, make_state, monkeypatch, cost_usd, expected_emitted,
):
    """A ContextOverflow becomes a normal termination and preserves the warning flag.

    Two scenarios:
      - cost_usd = 0.0  -> no soft warning was set up, budget_warning_emitted stays False;
      - cost_usd = 0.45 -> soft warning was triggered on this same step, so the
        flag is True and is preserved on the overflow path.
    """
    import services.agent.context.context_window as cw_mod

    monkeypatch.setattr(cw_mod, "MAX_ESTIMATED_CHARS", 100)

    llm = FakeLLM()
    ctx = make_context(llm=llm)
    state = make_state(ctx)
    state["budget"].cost_usd = cost_usd

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.CONTEXT_OVERFLOW
    assert result["budget_warning_emitted"] is expected_emitted
    assert llm.calls == []

    assert any(r.get("kind") == "context_overflow" for r in trace.records())


async def test_decide_calls_build_context_once(trace, make_context, make_state, monkeypatch):
    """build_context is invoked exactly once per decide_node call.

    Guards against a future regression that re-splits context assembly
    into two calls (the footgun the refactor removed).
    """
    import services.agent.agent_graph.nodes as nodes_mod

    real_build = nodes_mod.build_context
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(nodes_mod, "build_context", spy)

    ctx = make_context()
    state = make_state(ctx)

    await decide_node(state, FakeRuntime(ctx))

    assert len(calls) == 1


async def test_decide_passes_history_to_build_context(trace, make_context, make_state, monkeypatch):
    """state['history'] reaches build_context by reference, not as a copy.

    build_context is called positionally: (system_prompt, warning, history).
    """
    import services.agent.agent_graph.nodes as nodes_mod

    real_build = nodes_mod.build_context
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(nodes_mod, "build_context", spy)

    ctx = make_context()
    state = make_state(ctx)

    await decide_node(state, FakeRuntime(ctx))

    assert len(calls) == 1
    args, _ = calls[0]
    assert args[2] is state["history"]


async def test_decide_skips_build_context_when_steps_exhausted(trace, make_context, make_state, monkeypatch):
    """When the step budget is exhausted, build_context is not called at all.

    No wasted work: the short-circuit in _check_budget fires before context
    assembly.
    """
    import services.agent.agent_graph.nodes as nodes_mod

    called = []
    monkeypatch.setattr(nodes_mod, "build_context", lambda *a, **kw: called.append(1))

    ctx = make_context(max_steps=1)
    state = make_state(ctx)
    state["budget"].steps = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_STEPS
    assert called == []


async def test_decide_forwards_context_window_to_request_llm(trace, make_context, make_state, monkeypatch):
    """build_context output is forwarded to _request_llm unchanged.

    Regression guard: after the refactor, the effective system prompt
    (task prompt + optional budget warning) must travel through
    ``context_window.system`` rather than being reconstructed inside
    ``_request_llm``. If someone re-introduces ``chat(..., system_prompt=ctx.task.system_prompt)``,
    the warning is silently lost — this test catches that.
    """
    import services.agent.agent_graph.nodes as nodes_mod

    real_request = nodes_mod._request_llm
    seen = {}

    async def spy(*, llm, tools, trace, budget, context_window):
        seen["cw"] = context_window
        return await real_request(
            llm=llm, tools=tools, trace=trace, budget=budget, context_window=context_window,
        )

    monkeypatch.setattr(nodes_mod, "_request_llm", spy)

    ctx = make_context()
    state = make_state(ctx)
    state["budget"].cost_usd = 0.45  # 90% of the default 0.50 cost budget -> soft warning

    await decide_node(state, FakeRuntime(ctx))

    assert isinstance(seen["cw"], AgentContextWindow)
    assert "WARNING: more than 80%" in seen["cw"].system
