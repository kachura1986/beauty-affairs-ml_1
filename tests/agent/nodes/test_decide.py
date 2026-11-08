"""DECIDE stage tests: _check_budget, _request_llm, decide_node."""
import logging
import time

import pytest

from services.agent.agent_graph.nodes import decide_node
from services.agent.agent_graph.nodes.decide import _check_budget, _request_llm
from services.agent.context import AgentContextWindow
from services.agent.domain import (
    AgentPolicy,
    Budget,
    BudgetLimits,
    TerminationReason,
)
from services.agent.trace import Trace

from _testing import (
    _FakeAPIStatusError,
    FakeLLM,
    FakeRuntime,
    internal_server_error,
    make_context_window,
    rate_limit_error,
    reply,
)


# ---------------------------------------------------------------------------
# _check_budget
# ---------------------------------------------------------------------------

def test_check_budget_steps_exhausted():
    """Exhausted step budget yields BUDGET_STEPS."""
    budget = Budget(limits=BudgetLimits(max_steps=1, max_seconds=10, max_tokens=10, max_cost_usd=1))
    budget.steps = 1
    assert _check_budget(budget) == TerminationReason.BUDGET_STEPS


def test_check_budget_time_exceeded():
    """Expired wall-clock budget yields BUDGET_TIME even when steps remain."""
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=0.001, max_tokens=10, max_cost_usd=1))
    budget.started = time.monotonic() - 10
    assert _check_budget(budget) == TerminationReason.BUDGET_TIME


def test_check_budget_ok():
    """Fresh budget with all limits intact yields None."""
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
    """Successful call increments steps and records tokens/cost."""
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
    """chat() and trace.step() must receive the same messages object — otherwise window_hash no longer matches what was sent."""
    import services.agent.agent_graph.nodes.decide as decide_mod

    seen = {}
    real_step = decide_mod.Trace.step

    def spy_step(self, n, window, reply, budget, latency_ms, *, system_prompt):
        seen["window"] = window
        return real_step(self, n, window, reply, budget, latency_ms, system_prompt=system_prompt)

    monkeypatch.setattr(decide_mod.Trace, "step", spy_step)

    messages = [{"role": "user", "content": "hi"}]
    cw = make_context_window(system="custom-system", messages=messages)
    llm = FakeLLM(reply(text="ok"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, context_window=cw)

    assert llm.calls[0]["messages"] is cw.messages
    assert seen["window"] is cw.messages
    assert seen["window"] is llm.calls[0]["messages"]


async def test_request_llm_error_path_does_not_copy_messages(trace, make_context, monkeypatch):
    """Error path must also pass the original messages object to chat() — a defensive copy would break identity."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1,
    ))
    messages = [{"role": "user", "content": "hi"}]
    cw = make_context_window(system="sys", messages=messages)

    seen = {}
    real_error = type(ctx.trace).error

    def spy_error(self, *, step, kind, message, latency_ms, budget):
        seen["kind"] = kind
        seen["step"] = step
        return real_error(self, step=step, kind=kind, message=message,
                          latency_ms=latency_ms, budget=budget)

    monkeypatch.setattr(type(ctx.trace), "error", spy_error)

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=cw,
    )

    assert seen["kind"] == "llm_error"
    assert seen["step"] == budget.steps  # step was incremented before the call
    assert llm.calls[0]["messages"] is cw.messages


async def test_request_llm_timeout_returns_budget_time(trace, make_context, monkeypatch):
    """_attempt TOCTOU guard short-circuits before chat() when the wall-clock budget expired.

    Pins elapsed() call count: run preflight + _attempt guard = 2; trace.error(snapshot) is
    not reached. A refactor that adds a budget check must update this count deliberately.
    """
    llm = FakeLLM(errors=[TimeoutError("slow")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10, max_seconds=1.0, max_tokens=1000, max_cost_usd=1,
    ))

    elapsed_calls: list[int] = []
    times = iter([0.0, 2.0, 2.0])

    def fake_elapsed(self):
        elapsed_calls.append(1)
        return next(times, 2.0)

    monkeypatch.setattr(Budget, "elapsed", fake_elapsed)

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TIME
    assert len(llm.calls) == 0  # TOCTOU guard: chat is never called
    assert len(elapsed_calls) == 2


async def test_request_llm_llm_error(trace, make_context):
    """LLM exception with remaining time yields LLM_ERROR."""
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
    """Steps are incremented even if the LLM call fails — the step was started."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert budget.steps == 1


async def test_request_llm_without_chat_method_returns_llm_error(trace, make_context):
    """LLM object lacking a chat method yields LLM_ERROR."""

    class NoChatLLM:
        model_name = "fake"

        def is_transient_error(self, exc: Exception) -> bool:
            # Mirror BaseLLMModel default so the no-chat ValueError classifies as permanent.
            return isinstance(exc, (ConnectionError, TimeoutError))

    ctx = make_context(llm=NoChatLLM())
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR


async def test_request_llm_exceeded_after_successful_call(trace, make_context):
    """Successful call that pushes usage over the limit yields BUDGET_TOKENS but preserves the reply."""
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


async def test_request_llm_timeout_capped_by_default(trace, make_context, monkeypatch):
    """LLM call timeout is capped by the default when remaining budget is larger."""
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
    """Token budget below MIN_EXPECTED_TOKENS short-circuits before chat() (strict '<')."""
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
    """Exactly MIN_EXPECTED_TOKENS remaining must pass through to chat() (strict '<')."""
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
    """Pre-flight rejection is recorded in the trace with kind='preflight_tokens'."""
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


async def test_request_llm_precondition_raises_on_exhausted_steps(trace, make_context):
    """LLMStepService raises RuntimeError on exhausted steps — an explicit raise survives ``python -O``, an assert does not."""
    llm = FakeLLM(reply(text="never"))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=1, max_seconds=10, max_tokens=1000, max_cost_usd=1,
    ))
    budget.steps = 1  # exhausted

    with pytest.raises(RuntimeError, match='precondition violated'):
        await _request_llm(
            llm=ctx.llm,
            tools=ctx.tools,
            trace=ctx.trace,
            budget=budget,
            context_window=make_context_window(),
        )

    assert llm.calls == []


# ---------------------------------------------------------------------------
# decide_node
# ---------------------------------------------------------------------------

async def test_decide_returns_budget_steps_when_no_steps_left(trace, make_context, make_state):
    """Exhausted step budget short-circuits DECIDE before the LLM is called."""
    llm = FakeLLM()
    ctx = make_context(llm=llm, max_steps=1)
    state = make_state(ctx)
    state["budget"].steps = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_STEPS
    assert result.get("needs_finalization") is not True
    assert llm.calls == []


async def test_decide_raises_if_llm_returns_nothing(trace, make_context, make_state, monkeypatch):
    """A (None, None) result from _request_llm is a contract violation."""
    import services.agent.agent_graph.nodes.decide as decide_mod
    async def fake_request_llm(**kwargs):
        return None, None

    monkeypatch.setattr(decide_mod, "_request_llm", fake_request_llm)

    ctx = make_context()
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="no reply"):
        await decide_node(state, FakeRuntime(ctx))


async def test_decide_empty_response_increments_and_adds_observation(trace, make_context, make_state):
    """Empty LLM reply increments the empty counter and appends an observation."""
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
    """Text-only reply becomes pending_answer and resets the empty counter."""
    llm = FakeLLM(reply(text="final answer"))
    ctx = make_context(llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_answer"] == "final answer"
    assert "pending_tool_calls" not in result
    assert result["consecutive_empty_steps"] == 0
    assert "termination_reason" not in result


async def test_decide_tool_calls_sets_pending_tool_calls_and_history(trace, make_context, make_state):
    """Reply with tool_calls becomes pending and is echoed into history."""
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
    """Tool-call reply overwrites a stale pending_answer from a previous step."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    ctx = make_context(llm=llm)
    state = make_state(ctx)
    state["pending_answer"] = "stale"

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_answer"] is None


# ---------------------------------------------------------------------------
# decide_node: budget reason with a preserved reply
# ---------------------------------------------------------------------------

async def test_decide_preserves_answer_when_budget_exceeded_after_final_text(
        trace, make_context, make_state,
):
    """A final-text reply that pushes usage over the token budget must become `answer`, not be dropped."""
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
    """A tool_call reply at budget exhaustion is not a final answer — the model asked to continue."""
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
    """Tool_calls make a reply non-final even when it also carries text — `answer` must stay absent."""
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
# ---------------------------------------------------------------------------

async def test_decide_preflight_budget_does_not_set_answer(trace, make_context, make_state):
    """A (None, reason) from the preflight path must leave `answer` absent without raising."""
    from core.config import settings

    llm = FakeLLM(reply(text="never"))
    ctx = make_context(llm=llm, max_tokens=settings.MIN_EXPECTED_TOKENS - 1)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TOKENS
    assert "answer" not in result
    assert llm.calls == []


async def test_decide_llm_error_does_not_set_answer(trace, make_context, make_state):
    """A failed LLM call does not qualify as an answer — the preserved-reply contract applies only to the success path."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.LLM_ERROR
    assert "answer" not in result


async def test_decide_emits_budget_warning_once(trace, make_context, make_state):
    """Soft budget warning sets the emitted flag exactly once and is passed to LLM."""
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
    """ContextOverflow becomes a normal termination and preserves the warning flag (both pre-set and set on this step)."""
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
    """build_context is invoked exactly once per decide_node call — guards against re-splitting context assembly."""
    import services.agent.agent_graph.nodes.decide as decide_mod

    real_build = decide_mod.build_context
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(decide_mod, "build_context", spy)

    ctx = make_context()
    state = make_state(ctx)

    await decide_node(state, FakeRuntime(ctx))

    assert len(calls) == 1


async def test_decide_passes_history_to_build_context(trace, make_context, make_state, monkeypatch):
    """state['history'] reaches build_context by reference, not as a copy."""
    import services.agent.agent_graph.nodes.decide as decide_mod

    real_build = decide_mod.build_context
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(decide_mod, "build_context", spy)

    ctx = make_context()
    state = make_state(ctx)

    await decide_node(state, FakeRuntime(ctx))

    assert len(calls) == 1
    args, _ = calls[0]
    assert args[2] is state["history"]


async def test_decide_skips_build_context_when_steps_exhausted(trace, make_context, make_state, monkeypatch):
    """Step-budget short-circuit fires before context assembly — build_context is never called."""
    import services.agent.agent_graph.nodes.decide as decide_mod

    called = []
    monkeypatch.setattr(decide_mod, "build_context", lambda *a, **kw: called.append(1))

    ctx = make_context(max_steps=1)
    state = make_state(ctx)
    state["budget"].steps = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_STEPS
    assert called == []


async def test_decide_forwards_context_window_to_request_llm(trace, make_context, make_state, monkeypatch):
    """build_context output is forwarded to _request_llm unchanged — a reconstructed system prompt would silently drop the warning."""
    import services.agent.agent_graph.nodes.decide as decide_mod

    real_request = decide_mod._request_llm
    seen = {}

    async def spy(*, llm, tools, trace, budget, context_window):
        seen["cw"] = context_window
        return await real_request(
            llm=llm, tools=tools, trace=trace, budget=budget, context_window=context_window,
        )

    monkeypatch.setattr(decide_mod, "_request_llm", spy)

    ctx = make_context()
    state = make_state(ctx)
    state["budget"].cost_usd = 0.45  # 90% of the default 0.50 cost budget -> soft warning

    await decide_node(state, FakeRuntime(ctx))

    assert isinstance(seen["cw"], AgentContextWindow)
    assert "WARNING: more than 80%" in seen["cw"].system


# ---------------------------------------------------------------------------
# Error classification in the trace (Phase A observability)
# ---------------------------------------------------------------------------

async def test_request_llm_transient_kind_in_trace(trace, make_context, fast_retry):
    """Transient error (429) is recorded as kind='llm_transient'.

    Both attempts fail transiently; the second error keeps the retry from hitting
    FakeLLM's empty reply list (IndexError). fast_retry removes the real backoff.
    """
    llm = FakeLLM(errors=[rate_limit_error(), rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR
    kinds = [r.get("kind") for r in trace.records() if r.get("t") == "error"]
    assert "llm_transient" in kinds
    assert "llm_error" not in kinds


async def test_request_llm_permanent_kind_in_trace(trace, make_context):
    """Permanent LLM error (RuntimeError) is recorded as kind='llm_error'."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR
    kinds = [r.get("kind") for r in trace.records() if r.get("t") == "error"]
    assert "llm_error" in kinds
    assert "llm_transient" not in kinds


# ---------------------------------------------------------------------------
# Retry (Phase B1)
# ---------------------------------------------------------------------------

async def test_request_llm_retries_on_429(trace, make_context, fast_retry):
    """Transient 429 is retried once; retry success returns the reply with steps=2."""
    llm = FakeLLM(reply(text="ok"), errors=[rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert reason is None
    assert result is not None and result.text == "ok"
    assert budget.steps == 2

    records = trace.records()
    error_kinds = [r.get("kind") for r in records if r.get("t") == "error"]
    assert "llm_transient" in error_kinds
    assert "llm_retry" in error_kinds
    step_records = [r for r in records if r.get("t") == "step"]
    assert len(step_records) == 1
    assert step_records[0]["step"] == 2


async def test_request_llm_retries_on_503(trace, make_context, fast_retry):
    """Transient 503 is retried once; retry success returns the reply with steps=2."""
    llm = FakeLLM(reply(text="ok"), errors=[internal_server_error(503)])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert reason is None
    assert result is not None and result.text == "ok"
    assert budget.steps == 2


async def test_request_llm_retries_at_most_once(trace, make_context, fast_retry):
    """Even with plenty of step budget, a failed retry is terminal — one retry is the maximum."""
    llm = FakeLLM(errors=[rate_limit_error(), rate_limit_error(), rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR
    assert budget.steps == 2  # not 3 — the failed retry does not trigger another retry
    assert len(llm.calls) == 2
    retries = [r for r in trace.records() if r.get("kind") == "llm_retry"]
    assert len(retries) == 1


async def test_request_llm_no_retry_on_400(trace, make_context):
    """Permanent 400 is not retried, regardless of the remaining step budget."""
    llm = FakeLLM(errors=[_FakeAPIStatusError(400)])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR
    assert budget.steps == 1
    kinds = [r.get("kind") for r in trace.records() if r.get("t") == "error"]
    assert "llm_error" in kinds
    assert "llm_retry" not in kinds


async def test_request_llm_no_retry_when_max_steps_1(trace, make_context):
    """Transient error with max_steps=1 emits llm_transient, no llm_retry, returns LLM_ERROR."""
    llm = FakeLLM(errors=[rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=1, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR
    assert budget.steps == 1
    kinds = [r.get("kind") for r in trace.records() if r.get("t") == "error"]
    assert "llm_transient" in kinds
    assert "llm_retry" not in kinds


async def test_request_llm_no_retry_when_time_exhausted(trace, make_context, monkeypatch):
    """No retry when the wall-clock budget expires between the attempt and the except branch."""
    llm = FakeLLM(errors=[rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    # Preflight + _attempt guard see elapsed=0.0 (pass); except branch sees 15.0
    # with max_seconds=10 → remaining=0 → BUDGET_TIME, no retry.
    times = iter([0.0, 0.0, 15.0])

    def fake_elapsed(self):
        return next(times, 15.0)

    monkeypatch.setattr(Budget, "elapsed", fake_elapsed)

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TIME
    kinds = [r.get("kind") for r in trace.records() if r.get("t") == "error"]
    assert "llm_retry" not in kinds


async def test_request_llm_no_retry_when_tokens_insufficient(trace, make_context, monkeypatch):
    """_can_retry rejects a retry when the token budget shrank below MIN_EXPECTED_TOKENS.

    ``remaining_tokens()`` is called twice: run preflight (call #1, pass) and
    ``_can_retry`` (call #2, reject). The per-call monkeypatch exercises the
    token branch of ``_can_retry`` without a production-code change.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="never"), errors=[rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    call_count = [0]

    def fake_remaining_tokens(self):
        call_count[0] += 1
        if call_count[0] == 1:  # run preflight: pass
            return settings.MIN_EXPECTED_TOKENS
        return settings.MIN_EXPECTED_TOKENS - 1  # _can_retry: reject

    monkeypatch.setattr(Budget, "remaining_tokens", fake_remaining_tokens)

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert reason == TerminationReason.LLM_ERROR
    assert len(llm.calls) == 1  # no retry
    assert not any(r.get("kind") == "llm_retry" for r in trace.records())


async def test_request_llm_retry_consumes_step(trace, make_context, fast_retry):
    """A successful retry consumes a second step — budget.steps reflects physical invocations."""
    llm = FakeLLM(reply(text="ok"), errors=[rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=2, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert reason is None
    assert result is not None and result.text == "ok"
    assert budget.steps == 2


async def test_request_llm_retry_does_not_exceed_max_steps(trace, make_context, fast_retry):
    """With max_steps=2, a failed retry terminates at steps=2 — no third attempt."""
    llm = FakeLLM(errors=[rate_limit_error(), rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=2, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR
    assert budget.steps == 2
    assert len(llm.calls) == 2


async def test_request_llm_retry_records_real_latency(trace, make_context, fast_retry):
    """llm_retry.latency_ms reflects the first attempt's real duration (raise_after sleeps before raising)."""
    llm = FakeLLM(reply(text="ok"), errors=[rate_limit_error()], raise_after=0.02)
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    retry_events = [r for r in trace.records() if r.get("kind") == "llm_retry"]
    assert len(retry_events) == 1
    assert retry_events[0]["latency_ms"] >= 15


async def test_request_llm_retry_records_separate_steps(trace, make_context, fast_retry):
    """llm_transient records the first attempt's step; llm_retry records the retry's step — they differ."""
    llm = FakeLLM(reply(text="ok"), errors=[rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    records = trace.records()
    transient = [r for r in records if r.get("kind") == "llm_transient"]
    retry = [r for r in records if r.get("kind") == "llm_retry"]
    assert len(transient) == 1
    assert len(retry) == 1
    assert transient[0]["step"] == 1
    assert retry[0]["step"] == 2


async def test_request_llm_retry_also_emits_llm_transient(trace, make_context, fast_retry):
    """llm_transient is emitted for every transient failure — including the retry's failure.

    Pins Z2: introducing llm_retry must not silence llm_transient on either attempt;
    dashboards counting llm_transient lose nothing.
    """
    llm = FakeLLM(errors=[rate_limit_error(), rate_limit_error()])
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
        context_window=make_context_window(),
    )

    records = trace.records()
    transient = [r for r in records if r.get("kind") == "llm_transient"]
    retry = [r for r in records if r.get("kind") == "llm_retry"]
    assert len(transient) == 2  # original failure + retry failure
    assert len(retry) == 1


# ---------------------------------------------------------------------------
# Regression: _attempt() failure isolation (#1)
# ---------------------------------------------------------------------------

async def test_attempt_trace_step_raises_does_not_double_count(
        trace, make_context, monkeypatch, caplog,
):
    """A trace.step() failure after a successful chat() is best-effort: no retry, no double-billing.

    Regression for #1: trace.step() used to be inside run()'s retry-scope, so a
    ConnectionError from trace.step() was classified as transient and triggered a second
    chat() call, doubling token/cost accounting.
    """

    def raise_conn(*args, **kwargs):
        raise ConnectionError("trace io failed")

    monkeypatch.setattr(Trace, "step", raise_conn)

    llm = FakeLLM(reply(text="ok", in_tokens=5, out_tokens=3, cost_usd=0.001))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    with caplog.at_level(logging.ERROR):
        result, reason = await _request_llm(
            llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
            context_window=make_context_window(),
        )

    assert len(llm.calls) == 1  # no retry — chat() called exactly once
    assert budget.steps == 1
    assert budget.tokens == 8  # not 16 — accounting ran once
    assert budget.cost_usd == pytest.approx(0.001)
    assert result is not None and result.text == "ok"
    assert reason is None  # run continues normally

    assert any("Failed to write step trace" in r.message for r in caplog.records)

    kinds = [r.get("kind") for r in trace.records() if r.get("t") == "error"]
    assert "llm_retry" not in kinds
    assert "llm_error" not in kinds


async def test_attempt_record_llm_raises_does_not_double_count(
        trace, make_context, monkeypatch, caplog,
):
    """A budget.record_llm() failure after a successful chat() hard-fails with LLM_ERROR, no retry.

    Regression for #1: record_llm() used to be inside run()'s retry-scope, so a
    transient-shaped exception from record_llm() triggered a second chat() call, doubling
    token/cost accounting. The already-received reply is intentionally dropped: returning
    it as an answer would produce an inconsistent trace (answer_chars > 0, tokens = 0).
    """

    def raise_conn(*args, **kwargs):
        raise ConnectionError("accounting failed")

    # Budget is a pydantic model with validate_assignment=True: patch the class, not the
    # instance. The test holds a single Budget, so a class patch is safe.
    monkeypatch.setattr(Budget, "record_llm", raise_conn)

    llm = FakeLLM(reply(text="ok", in_tokens=5, out_tokens=3, cost_usd=0.001))
    ctx = make_context(llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=3, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    with caplog.at_level(logging.ERROR):
        result, reason = await _request_llm(
            llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget,
            context_window=make_context_window(),
        )

    assert len(llm.calls) == 1  # no retry — chat() called exactly once
    assert budget.steps == 1
    assert budget.tokens == 0  # accounting failed before incrementing self.tokens
    assert result is None  # reply dropped intentionally (see _attempt docstring)
    assert reason == TerminationReason.LLM_ERROR

    assert any("Failed to record LLM usage" in r.message for r in caplog.records)

    llm_error_records = [r for r in trace.records() if r.get("kind") == "llm_error"]
    assert len(llm_error_records) == 1
    assert llm_error_records[0]["message"] == "ConnectionError"
    assert not any(r.get("kind") == "llm_retry" for r in trace.records())
