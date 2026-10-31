import pytest
import time
from pathlib import Path
from pydantic import BaseModel, ConfigDict

from services.agent.agent_graph.nodes import (
    _check_budget,
    _request_llm,
    decide_node,
    execute_node,
    verify_node,
    finish_node,
)
from services.agent.agent_graph.routing import (
    route_after_decide,
    route_after_execute,
    route_after_verify,
)
from services.agent.agent_graph.state import AgentContext, create_initial_state
from services.agent.domain import (
    Budget,
    BudgetLimits,
    AgentPolicy,
    Task,
    TerminationReason,
    AgentResult,
)
from services.agent.tools import Tool, ToolRegistry
from services.agent.trace import Trace
from services.llm import Reply


# ---------------------------------------------------------------------------
# Fake infrastructure
# ---------------------------------------------------------------------------

class FakeLLM:
    """Protocol-compatible LLM: returns canned replies or raises scripted errors."""

    def __init__(self, *replies: Reply, errors: list[Exception] | None = None):
        self._replies = list(replies)
        self._errors = list(errors or [])
        self.calls: list[dict] = []
        self.model_name = "fake"

    async def chat(self, messages, tools=None, timeout=None):
        self.calls.append({"messages": messages, "tools": tools, "timeout": timeout})
        if self._errors:
            raise self._errors.pop(0)
        return self._replies.pop(0)


class FakeRuntime:
    """Minimal Runtime[AgentContext] stand-in."""

    def __init__(self, context: AgentContext):
        self.context = context


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "ok"


async def echo_fn(value: str = "ok"):
    return value


async def failing_fn(**kwargs):
    raise RuntimeError("internal boom")


async def timeout_fn(**kwargs):
    raise TimeoutError("temporary timeout")


def reply(text="ok", tool_calls=None, in_tokens=0, out_tokens=0, cost_usd=0.0, model="fake"):
    """Helper to reduce noise and make tests resilient to Reply changes."""
    return Reply(
        text=text,
        tool_calls=tool_calls or [],
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost_usd=cost_usd,
        model=model,
    )


def make_registry(*extra_tools: Tool) -> ToolRegistry:
    tools = [
        Tool(name="echo", description="echo", fn=echo_fn, args_model=EchoArgs),
        *extra_tools,
    ]
    return ToolRegistry(tools, default_timeout_seconds=30.0, max_output_chars=15000)


def make_context(
        trace: Trace,
        llm: FakeLLM | None = None,
        tools: ToolRegistry | None = None,
        postcondition=None,
        policy: AgentPolicy | None = None,
        max_steps: int = 16,
        max_seconds: float = 300.0,
        max_tokens: int = 200_000,
        max_cost_usd: float = 0.50,
        system_prompt: str = "sys",
) -> AgentContext:
    task = Task(
        system_prompt=system_prompt,
        initial_messages=[{"role": "user", "content": "hi"}],
        postcondition=postcondition,
        budget_limits=BudgetLimits(
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
        ),
    )
    return AgentContext(
        task=task,
        tools=tools or make_registry(),
        llm=llm or FakeLLM(reply(text="ok")),
        trace=trace,
        policy=policy or AgentPolicy(empty_limit=3, repeat_limit=3),
    )


def make_state(ctx: AgentContext):
    return create_initial_state(ctx.task)


@pytest.fixture
def trace(tmp_path: Path):
    t = Trace(tmp_path)
    yield t
    t.close()


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

async def test_request_llm_success(trace):
    """A successful LLM call increments steps and records tokens and cost."""
    llm = FakeLLM(reply(text="hello", in_tokens=5, out_tokens=3, cost_usd=0.001))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert reason is None
    assert result is not None and result.text == "hello"
    assert budget.steps == 1
    assert budget.tokens == 8
    assert budget.cost_usd == pytest.approx(0.001)
    assert llm.calls[0]["tools"] == ctx.tools.schemas()
    assert llm.calls[0]["timeout"] == pytest.approx(10.0, abs=0.1)


async def test_request_llm_timeout_returns_budget_time(trace, monkeypatch):
    """An LLM exception on an exhausted wall-clock budget yields BUDGET_TIME.

    ``_request_llm`` re-checks the remaining time after the exception, so a
    raised error must be classified as BUDGET_TIME rather than LLM_ERROR when
    the run has already run out of time during the call.
    """
    llm = FakeLLM(errors=[TimeoutError("slow")])
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=1.0, max_tokens=1000, max_cost_usd=1))

    # Emulate that the LLM call took longer than the remaining budget (1.0s).
    # Use a fallback to prevent StopIteration when budget.snapshot() calls elapsed() again.
    # Patch the method on the class: Budget uses validate_assignment=True, so an
    # instance-level setattr of a non-field attribute (such as a replacement
    # `elapsed`) is rejected by Pydantic. monkeypatch restores it after the test.
    times = iter([0.0, 2.0])
    monkeypatch.setattr(Budget, "elapsed", lambda self: next(times, 2.0))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TIME
    assert len(llm.calls) == 1


async def test_request_llm_llm_error(trace):
    """An LLM exception with remaining time yields LLM_ERROR."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR


async def test_request_llm_error_still_increments_steps(trace):
    """Steps are incremented even if the LLM call fails, as the step was started."""
    llm = FakeLLM(errors=[RuntimeError("boom")])
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert budget.steps == 1


async def test_request_llm_without_chat_method_returns_llm_error(trace):
    """An LLM object lacking a chat method yields LLM_ERROR."""

    class NoChatLLM:
        model_name = "fake"

    ctx = make_context(trace, llm=NoChatLLM())
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert result is None
    assert reason == TerminationReason.LLM_ERROR


async def test_request_llm_exceeded_after_successful_call(trace):
    """A successful call that pushes usage over the limit yields BUDGET_TOKENS."""
    llm = FakeLLM(reply(text="x", in_tokens=100, out_tokens=0, cost_usd=0))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1, max_cost_usd=1))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TOKENS


async def test_request_llm_no_steps_left(trace):
    """No steps left short-circuits before any LLM call is made."""
    llm = FakeLLM(reply(text="never"))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=1, max_seconds=10, max_tokens=1000, max_cost_usd=1))
    budget.steps = 1

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_STEPS
    assert llm.calls == []


async def test_request_llm_timeout_capped_by_remaining_budget(trace):
    """The LLM call timeout equals the remaining budget when it is below the cap.

    With the default LLM_DEFAULT_TIMEOUT_SECONDS (120s), a 10s budget must
    cap the call at ~10s rather than letting it run for the full default.
    """
    llm = FakeLLM(reply(text="hello"))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=10, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert llm.calls[0]["timeout"] == pytest.approx(10.0, abs=0.1)


async def test_request_llm_timeout_capped_by_default(trace, monkeypatch):
    """The LLM call timeout equals the default cap when the budget is larger.

    A 300s budget must not force a call to wait the whole 300s when
    LLM_DEFAULT_TIMEOUT_SECONDS is 5s: min(300, 5) == 5.
    """
    from core.config import settings
    monkeypatch.setattr(settings, "LLM_DEFAULT_TIMEOUT_SECONDS", 5.0)

    llm = FakeLLM(reply(text="hello"))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(max_steps=10, max_seconds=300, max_tokens=1000, max_cost_usd=1))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert llm.calls[0]["timeout"] == pytest.approx(5.0, abs=0.1)


async def test_request_llm_preflight_tokens_below_minimum(trace):
    """A token budget below MIN_EXPECTED_TOKENS short-circuits before chat().

    The check is strict '<', so a caller that lands exactly on the threshold
    must still be allowed to try (see the boundary test). A clearly
    insufficient budget must never reach the LLM.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="never"))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS - 1,
        max_cost_usd=1,
    ))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert result is None
    assert reason == TerminationReason.BUDGET_TOKENS
    assert llm.calls == []


async def test_request_llm_preflight_tokens_at_boundary_allowed(trace):
    """Exactly MIN_EXPECTED_TOKENS remaining is still allowed to call the LLM.

    The pre-flight uses strict '<', so the boundary value must pass through
    to chat() rather than terminating early.
    """
    from core.config import settings

    llm = FakeLLM(reply(text="ok"))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS,
        max_cost_usd=1,
    ))

    result, reason = await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    assert reason is None
    assert result is not None
    assert len(llm.calls) == 1


async def test_request_llm_preflight_tokens_emits_trace(trace):
    """The pre-flight rejection is recorded in the trace with kind='preflight_tokens'.

    Parses the JSONL file rather than substring-matching, so the check is
    independent of the trace's separators and field ordering.
    """
    import json
    from core.config import settings

    llm = FakeLLM(reply(text="never"))
    ctx = make_context(trace, llm=llm)
    budget = Budget(limits=BudgetLimits(
        max_steps=10,
        max_seconds=10,
        max_tokens=settings.MIN_EXPECTED_TOKENS - 1,
        max_cost_usd=1,
    ))

    await _request_llm(
        llm=ctx.llm, tools=ctx.tools, trace=ctx.trace, budget=budget, window=[]
    )

    records = [
        json.loads(line)
        for line in trace.path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert any(r.get("kind") == "preflight_tokens" for r in records)


# ---------------------------------------------------------------------------
# decide_node
# ---------------------------------------------------------------------------

async def test_decide_returns_budget_steps_when_no_steps_left(trace):
    """An exhausted step budget short-circuits DECIDE before the LLM is called."""
    llm = FakeLLM()
    ctx = make_context(trace, llm=llm, max_steps=1)
    state = make_state(ctx)
    state["budget"].steps = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_STEPS
    assert llm.calls == []


async def test_decide_raises_if_llm_returns_nothing(trace, monkeypatch):
    """A (None, None) result from _request_llm is a contract violation."""
    import services.agent.agent_graph.nodes as nodes_mod
    async def fake_request_llm(**kwargs):
        return None, None

    monkeypatch.setattr(nodes_mod, "_request_llm", fake_request_llm)

    ctx = make_context(trace)
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="no reply"):
        await decide_node(state, FakeRuntime(ctx))


async def test_decide_empty_response_increments_and_adds_observation(trace):
    """An empty LLM reply increments the empty counter and appends an observation."""
    llm = FakeLLM(reply(text="   "))
    ctx = make_context(trace, llm=llm, policy=AgentPolicy(empty_limit=2, repeat_limit=3))
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["consecutive_empty_steps"] == 1
    assert result["history"][0]["role"] == "user"
    assert "[observation]" in result["history"][0]["content"]
    assert "termination_reason" not in result
    assert result["budget"] is state["budget"]


async def test_decide_empty_response_exceeds_limit_returns_no_progress(trace):
    """Crossing the empty-response limit terminates with NO_PROGRESS."""
    llm = FakeLLM(reply(text=""))
    ctx = make_context(trace, llm=llm, policy=AgentPolicy(empty_limit=1, repeat_limit=3))
    state = make_state(ctx)
    state["consecutive_empty_steps"] = 1

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.NO_PROGRESS
    assert result["budget"] is state["budget"]


async def test_decide_text_answer_sets_pending_answer(trace):
    """A text-only reply becomes pending_answer and resets the empty counter."""
    llm = FakeLLM(reply(text="final answer"))
    ctx = make_context(trace, llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_answer"] == "final answer"
    assert "pending_tool_calls" not in result
    assert result["consecutive_empty_steps"] == 0
    assert "termination_reason" not in result


async def test_decide_tool_calls_sets_pending_tool_calls_and_history(trace):
    """A reply with tool_calls becomes pending and is echoed into history."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    ctx = make_context(trace, llm=llm)
    state = make_state(ctx)

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_tool_calls"] == [call]
    assert result["pending_answer"] is None
    assert result["history"][0]["role"] == "assistant"
    assert result["history"][0]["tool_calls"] == [call]
    assert "termination_reason" not in result


async def test_decide_tool_calls_clears_stale_pending_answer(trace):
    """A tool-call reply overwrites a stale pending_answer from a previous step."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    llm = FakeLLM(reply(text="", tool_calls=[call]))
    ctx = make_context(trace, llm=llm)
    state = make_state(ctx)
    state["pending_answer"] = "stale"

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["pending_answer"] is None


async def test_decide_emits_budget_warning_once(trace):
    """The soft budget warning sets the emitted flag exactly once and is passed to LLM."""
    llm = FakeLLM(reply(text="a"), reply(text="b"))
    ctx = make_context(trace, llm=llm, max_steps=10)
    state = make_state(ctx)
    state["budget"].cost_usd = 0.45  # 90% of the default 0.50 cost budget

    first = await decide_node(state, FakeRuntime(ctx))
    assert first["budget_warning_emitted"] is True
    assert any("WARNING: more than 80%" in m["content"] for m in llm.calls[0]["messages"])

    state["budget_warning_emitted"] = True
    second = await decide_node(state, FakeRuntime(ctx))
    assert second["budget_warning_emitted"] is True
    assert not any("WARNING: more than 80%" in m["content"] for m in llm.calls[1]["messages"])


async def test_decide_budget_warning_not_reemitted(trace):
    """Once emitted, the warning flag stays True on subsequent steps."""
    llm = FakeLLM(reply(text="a"))
    ctx = make_context(trace, llm=llm, max_steps=10)
    state = make_state(ctx)
    state["budget"].cost_usd = 0.45
    state["budget_warning_emitted"] = True

    result = await decide_node(state, FakeRuntime(ctx))

    assert result["budget_warning_emitted"] is True
    assert not any("WARNING: more than 80%" in m["content"] for m in llm.calls[0]["messages"])


# ---------------------------------------------------------------------------
# execute_node
# ---------------------------------------------------------------------------

async def test_execute_successful_tool_call(trace):
    """A successful call clears pending calls and appends a tool message."""
    call = {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
    ctx = make_context(trace)
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["pending_tool_calls"] == []
    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert "hi" in result["history"][0]["content"]
    assert result["action_counts"]


async def test_execute_irrecoverable_failed_action_skipped(trace):
    """A non-retryable failure is recorded, then skipped on the next attempt.

    The second execution must not invoke the tool again; instead it must
    append a structured already_failed tool message with the matching
    tool_call_id (an [observation] user message would break the tool_call
    contract — that was the old, buggy behaviour).
    """
    call = {"id": "c1", "name": "failing", "arguments": {}}
    registry = make_registry(
        Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
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


async def test_execute_repeated_action_terminates(trace):
    """Exceeding repeat_limit terminates with REPEATED_ACTION without calling the tool.

    The offending call still receives a matching tool message so that the
    assistant tool_calls stay paired in history; the loop must not `break`.
    """
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(trace, policy=AgentPolicy(empty_limit=3, repeat_limit=2))
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


async def test_execute_preflight_time_skips_tool(trace):
    """An already-expired wall-clock budget short-circuits before invoke().

    Between the LLM step and execute_node some time may pass, so the budget
    can expire before the first tool call. The tool must not be invoked, and
    the skipped call must still receive a paired tool message.
    """
    call = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(trace, max_seconds=0.001)
    state = make_state(ctx)
    state["budget"].started = time.monotonic() - 10
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.BUDGET_TIME
    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert result["history"][0]["tool_call_id"] == "c1"
    assert "was not executed" in result["history"][0]["content"]


async def test_execute_stops_after_budget_exceeded_between_calls(trace):
    """If the budget expires during the first tool, the second must not execute."""
    calls_made = []

    async def slow_fn(**kwargs):
        calls_made.append(1)
        state["budget"].started -= 10_000  # emulate time passing
        return "ok"

    registry = make_registry(
        Tool(name="slow", description="", fn=slow_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry, max_seconds=10)
    state = make_state(ctx)
    call1 = {"id": "c1", "name": "slow", "arguments": {}}
    call2 = {"id": "c2", "name": "echo", "arguments": {}}
    state["pending_tool_calls"] = [call1, call2]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(calls_made) == 1
    assert result["termination_reason"] == TerminationReason.BUDGET_TIME


async def test_execute_multiple_tool_calls_partial_failure(trace):
    """When one of several calls fails, all results are still appended to history."""
    ok = {"id": "c1", "name": "echo", "arguments": {"value": "ok"}}
    fail = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(
        Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [ok, fail]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["history"]) == 2
    assert result["history"][0]["role"] == "tool"
    assert result["history"][1]["role"] == "tool"
    assert result["failed_actions"]


async def test_execute_temporary_failure_does_not_add_to_failed_set(trace):
    """Temporary failures stay retryable and are not added to failed_actions."""
    call = {"id": "c1", "name": "timeout", "arguments": {}}
    registry = make_registry(
        Tool(name="timeout", description="", fn=timeout_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["failed_actions"] == []


async def test_execute_skips_duplicate_failed_action(trace):
    """A call whose fingerprint is already in failed_actions is skipped.

    The skipped call must still receive a tool message with the matching
    tool_call_id so the assistant tool_calls stay paired in history.
    """
    call = {"id": "c1", "name": "failing", "arguments": {}}
    registry = make_registry(
        Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
    state = make_state(ctx)
    fp = ToolRegistry.fingerprint(call)
    state["failed_actions"] = [fp]
    state["pending_tool_calls"] = [call]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["history"]) == 1
    assert result["history"][0]["role"] == "tool"
    assert result["history"][0]["tool_call_id"] == "c1"
    assert "previously failed irrecoverably" in result["history"][0]["content"]


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------

def test_verify_no_postcondition_done_unverified(trace):
    """No postcondition configured yields DONE_UNVERIFIED."""
    ctx = make_context(trace, postcondition=None)
    state = make_state(ctx)
    state["pending_answer"] = "answer"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.DONE_UNVERIFIED
    assert result["answer"] == "answer"
    assert result["pending_answer"] is None


def test_verify_postcondition_ok_done(trace):
    """A passing postcondition yields DONE."""
    ctx = make_context(trace, postcondition=lambda a: (True, "fine"))
    state = make_state(ctx)
    state["pending_answer"] = "good"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.DONE
    assert result["answer"] == "good"


def test_verify_postcondition_fails_adds_observation(trace):
    """A failing postcondition returns to DECIDE with an observation.

    The original answer is not accepted; instead the note from the
    postcondition is fed back to the model for another attempt.
    """
    ctx = make_context(trace, postcondition=lambda a: (False, "missing detail"))
    state = make_state(ctx)
    state["pending_answer"] = "bad"

    result = verify_node(state, FakeRuntime(ctx))

    assert "termination_reason" not in result
    assert result["history"][0]["role"] == "user"
    assert "Postcondition not satisfied" in result["history"][0]["content"]
    assert "missing detail" in result["history"][0]["content"]


def test_verify_postcondition_exception_returns_postcondition_error(trace):
    """A raising postcondition preserves the answer and reports POSTCONDITION_ERROR."""

    def broken(_):
        raise RuntimeError("boom")

    ctx = make_context(trace, postcondition=broken)
    state = make_state(ctx)
    state["pending_answer"] = "answer"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.POSTCONDITION_ERROR
    assert result["answer"] == "answer"


def test_verify_postcondition_returns_invalid_shape(trace):
    """Returning a non-tuple from postcondition yields POSTCONDITION_ERROR."""
    ctx = make_context(trace, postcondition=lambda a: "not a tuple")
    state = make_state(ctx)
    state["pending_answer"] = "x"

    result = verify_node(state, FakeRuntime(ctx))

    assert result["termination_reason"] == TerminationReason.POSTCONDITION_ERROR
    assert result["answer"] == "x"


def test_verify_without_pending_answer_raises(trace):
    """Reaching VERIFY without a pending answer is a contract violation."""
    ctx = make_context(trace)
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="VERIFY phase has no pending answer"):
        verify_node(state, FakeRuntime(ctx))


def test_verify_postcondition_note_in_observation(trace):
    """The postcondition note is embedded verbatim in the observation."""
    ctx = make_context(trace, postcondition=lambda a: (False, "note-XYZ"))
    state = make_state(ctx)
    state["pending_answer"] = "x"

    result = verify_node(state, FakeRuntime(ctx))

    assert "note-XYZ" in result["history"][0]["content"]


def test_verify_postcondition_fails_with_empty_note(trace):
    """An empty note from postcondition does not break the flow."""
    ctx = make_context(trace, postcondition=lambda a: (False, ""))
    state = make_state(ctx)
    state["pending_answer"] = "x"

    result = verify_node(state, FakeRuntime(ctx))

    assert "Postcondition not satisfied" in result["history"][0]["content"]


# ---------------------------------------------------------------------------
# finish_node
# ---------------------------------------------------------------------------

def test_finish_without_termination_reason_raises(trace):
    """Reaching FINISH without a termination reason is a contract violation."""
    ctx = make_context(trace)
    state = make_state(ctx)

    with pytest.raises(RuntimeError, match="Finish node requires a termination reason"):
        finish_node(state, FakeRuntime(ctx))


def test_finish_creates_agent_result(trace):
    """FINISH builds a stable AgentResult with reason, answer, and trace path."""
    ctx = make_context(trace)
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.DONE
    state["answer"] = "final"

    result = finish_node(state, FakeRuntime(ctx))

    agent_result = result["result"]
    assert isinstance(agent_result, AgentResult)
    assert agent_result.reason == TerminationReason.DONE
    assert agent_result.answer == "final"
    assert agent_result.trace == str(ctx.trace.path)


def test_finish_result_numeric_fields(trace):
    """FINISH accurately populates numeric fields in AgentResult."""
    ctx = make_context(trace)
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


def test_finish_without_answer(trace):
    """FINISH handles a missing answer gracefully."""
    ctx = make_context(trace)
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.BUDGET_TIME

    result = finish_node(state, FakeRuntime(ctx))

    assert result["result"].answer is None
    content = trace.path.read_text(encoding="utf-8")
    assert '"answer_chars": 0' in content


def test_finish_trace_finish_recorded(trace):
    """FINISH writes the final record to the trace file."""
    ctx = make_context(trace)
    state = make_state(ctx)
    state["termination_reason"] = TerminationReason.DONE_UNVERIFIED
    state["answer"] = "x"

    finish_node(state, FakeRuntime(ctx))

    content = trace.path.read_text(encoding="utf-8")
    assert '"t": "finish"' in content
    assert '"reason": "done_unverified"' in content


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", [
    ({"termination_reason": TerminationReason.DONE}, "finish"),
    ({"pending_tool_calls": [{"name": "x"}]}, "execute"),
    ({"pending_answer": "a"}, "verify"),
    ({}, "decide"),
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
# Additional edge cases
# ---------------------------------------------------------------------------

async def test_execute_multiple_calls_second_fails_irrecoverable(trace):
    """A second call failing irrecoverably still records both results."""
    ok = {"id": "c1", "name": "echo", "arguments": {"value": "ok"}}
    bad = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(
        Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [ok, bad]

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["failed_actions"]) == 1
    assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# Regression tests for the execute_node tool_call pairing invariant
# ---------------------------------------------------------------------------

async def test_regression_failed_set_same_batch(trace):
    """Two identical failed calls in one batch each get a matching tool message.

    Regression for the bug where a repeated failed call inside the same batch
    was answered with build_observation_message (role='user') instead of
    build_tool_message, leaving the assistant tool_calls unpaired.
    """
    call1 = {"id": "c1", "name": "failing", "arguments": {}}
    call2 = {"id": "c2", "name": "failing", "arguments": {}}
    registry = make_registry(
        Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
    state = make_state(ctx)
    state["pending_tool_calls"] = [call1, call2]

    result = await execute_node(state, FakeRuntime(ctx))

    assert result.get("termination_reason") is None
    assert len(result["history"]) == 2
    assert [m["role"] for m in result["history"]] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in result["history"]] == ["c1", "c2"]
    # First ran and failed; second was answered synthetically.
    assert "internal error" in result["history"][0]["content"]
    assert "previously failed irrecoverably" in result["history"][1]["content"]


async def test_regression_repeated_action_three_calls(trace):
    """Three identical calls with repeat_limit=1 produce three paired tool messages.

    Regression for the bug where exceeding repeat_limit issued a `break`,
    leaving the remaining calls without tool messages. Expected shape:
    call1 executes normally, call2 trips the limit, call3 is skipped.
    """
    call1 = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    call2 = {"id": "c2", "name": "echo", "arguments": {"value": "x"}}
    call3 = {"id": "c3", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(trace, policy=AgentPolicy(empty_limit=3, repeat_limit=1))
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


async def test_regression_budget_exceeded_mid_batch(trace):
    """Budget overrun after the first call leaves the remaining calls paired.

    Regression for the bug where hitting a hard budget mid-batch issued a
    `break`, leaving trailing tool_calls without matching tool messages. The
    two trailing calls must now carry skipped_due_to_termination results.
    """
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
    ctx = make_context(trace, tools=registry, max_seconds=10)
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


# ---------------------------------------------------------------------------
# Property: every pending tool call is paired with exactly one tool message
# ---------------------------------------------------------------------------

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
                [Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)],
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
async def test_property_pending_calls_are_all_paired(trace, calls, policy_kwargs, extra_tools):
    """Every pending call receives exactly one tool message with a matching id.

    This is the invariant that execute_node must preserve on any exit path.
    A failure here means a future refactor introduced a `break`, an early
    return, or a plain observation message in place of a tool message.
    """
    registry = make_registry(*extra_tools)
    ctx = make_context(trace, tools=registry, policy=AgentPolicy(**policy_kwargs))
    state = make_state(ctx)
    state["pending_tool_calls"] = list(calls)

    result = await execute_node(state, FakeRuntime(ctx))

    assert len(result["history"]) == len(calls)
    assert all(m["role"] == "tool" for m in result["history"])
    assert [m["tool_call_id"] for m in result["history"]] == [c["id"] for c in calls]


# ---------------------------------------------------------------------------
# Additional edge cases for the pairing invariant
# ---------------------------------------------------------------------------

async def test_execute_empty_batch_is_noop(trace):
    """An empty pending_tool_calls produces no history and no termination."""
    ctx = make_context(trace)
    state = make_state(ctx)
    state["pending_tool_calls"] = []

    result = await execute_node(state, FakeRuntime(ctx))

    assert result["history"] == []
    assert result["pending_tool_calls"] == []
    assert "termination_reason" not in result


async def test_regression_skipped_message_carries_reason(trace):
    """A call skipped after termination records the reason in its content.

    Together with trace-side visibility, this lets post-mortem and regression
    tests distinguish a real execution from a synthetic skip. Note that the
    call which *trips* the limit carries repeated_action_result; only the
    calls after it go through skipped_due_to_termination_result, which is
    the message that embeds the termination reason verbatim.
    """
    call1 = {"id": "c1", "name": "echo", "arguments": {"value": "x"}}
    call2 = {"id": "c2", "name": "echo", "arguments": {"value": "x"}}
    call3 = {"id": "c3", "name": "echo", "arguments": {"value": "x"}}
    ctx = make_context(trace, policy=AgentPolicy(empty_limit=3, repeat_limit=1))
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


async def test_execute_failed_set_persists_across_batches(trace):
    """failed_actions recorded in one batch are honoured on the next batch.

    Verifies that the failed-set state is properly threaded through the graph
    state and produces a synthetic response instead of re-invoking the tool.
    """
    call = {"id": "c1", "name": "failing", "arguments": {}}
    registry = make_registry(
        Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)
    )
    ctx = make_context(trace, tools=registry)
    state = make_state(ctx)

    state["pending_tool_calls"] = [call]
    first = await execute_node(state, FakeRuntime(ctx))
    assert first["failed_actions"]

    # Second batch carries the recorded fingerprint forward.
    state["pending_tool_calls"] = [call]
    state["failed_actions"] = first["failed_actions"]
    second = await execute_node(state, FakeRuntime(ctx))

    assert second["history"][0]["role"] == "tool"
    assert second["history"][0]["tool_call_id"] == "c1"
    assert "previously failed irrecoverably" in second["history"][0]["content"]
