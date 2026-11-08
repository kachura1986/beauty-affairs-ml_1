"""Tests for AgentService concurrency limiting."""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.agent.domain import (
    AgentPolicy,
    AgentResult,
    BudgetLimits,
    Finalization,
    TerminationReason,
)
from services.agent.tools import ToolRegistry
from services.llm import Reply
from use_cases.agent_service import AgentService


def _fake_result() -> AgentResult:
    """Build a real AgentResult so isinstance checks downstream do not break."""
    return AgentResult(
        run_id="test",
        reason=TerminationReason.DONE,
        answer="ok",
        steps=0,
        tokens=0,
        cost_usd=0.0,
        seconds=0.0,
        history_len=0,
        trace="/tmp/test",
        finalization=Finalization.NONE,
    )


def _msg(role: str = "user", content: str = "hi") -> SimpleNamespace:
    """Build a Message-shaped object with attribute access (Protocol-compatible)."""
    return SimpleNamespace(role=role, content=content)


class _GatedLLM:
    """Protocol-compatible LLM stub whose first call blocks until released.

    The second and later calls return immediately. Used to hold the
    semaphore slot for a controlled amount of time, so a queued run can
    be observed against its own ``Budget.max_seconds``.
    """

    model_name = "fake"

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._calls = 0

    async def chat(
            self,
            messages,
            tools=None,
            timeout=None,
            system_prompt=None,
    ) -> Reply:
        self._calls += 1
        if self._calls == 1:
            self.entered.set()
            await self.release.wait()
        return Reply(text="ok")


class _DelayLLM:
    """LLM stub whose chat() sleeps for ``delay`` seconds.

    Provides the async yield that makes concurrent execution observable: without a real await inside
    the critical section, even a buggy concurrency would appear serialized.
    """

    model_name = "fake"

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay

    async def chat(
            self,
            messages,
            tools=None,
            timeout=None,
            system_prompt=None,
    ) -> Reply:
        await asyncio.sleep(self._delay)
        return Reply(text="ok")


@pytest.fixture
def service_kwargs(tmp_path):
    return dict(
        llm=MagicMock(),
        tools=MagicMock(),
        system_prompt="sys",
        runs_dir=tmp_path,
        policy=AgentPolicy(empty_limit=1, repeat_limit=1),
        budget_limits=BudgetLimits(max_steps=1, max_seconds=10, max_tokens=1000, max_cost_usd=0),
    )


async def test_semaphore_limits_concurrent_runs(service_kwargs, monkeypatch):
    """At most MAX_CONCURRENT_RUNS runs execute simultaneously; the rest queue on the semaphore."""
    import use_cases.agent_service as agent_service_mod

    concurrent = 0
    peak = 0
    gate = asyncio.Event()
    two_entered = asyncio.Event()

    async def fake_run_graph(*args, **kwargs):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        if concurrent == 2:
            two_entered.set()
        try:
            await gate.wait()
            return _fake_result()
        finally:
            concurrent -= 1

    monkeypatch.setattr(agent_service_mod, "run_graph", fake_run_graph)

    semaphore = asyncio.Semaphore(2)
    service = AgentService(**service_kwargs, semaphore=semaphore)

    tasks = [
        asyncio.create_task(service.run([_msg()]))
        for _ in range(4)
    ]

    # Deterministically wait until two runs are inside run_graph.
    await asyncio.wait_for(two_entered.wait(), timeout=5.0)
    assert peak == 2

    gate.set()
    await asyncio.gather(*tasks)

    # Peak stayed at 2 across the whole lifetime.
    assert peak == 2


async def test_validation_errors_do_not_consume_slot(service_kwargs, monkeypatch):
    """Validation raises BEFORE semaphore acquisition; a bad request cannot starve a good one."""
    import use_cases.agent_service as agent_service_mod

    calls = []

    async def fake_run_graph(*args, **kwargs):
        calls.append(1)
        return _fake_result()

    monkeypatch.setattr(agent_service_mod, "run_graph", fake_run_graph)

    semaphore = asyncio.Semaphore(1)
    service = AgentService(**service_kwargs, semaphore=semaphore)

    with pytest.raises(ValueError, match="Messages cannot be empty"):
        await service.run([])
    with pytest.raises(ValueError, match="only 'user' and 'assistant' are allowed"):
        await service.run([_msg(role="system", content="bad")])

    # _value is private, but in a test it proves the slot was never acquired.
    assert semaphore._value == 1

    await service.run([_msg()])
    assert len(calls) == 1


async def test_queue_wait_not_counted_in_budget(service_kwargs):
    """
    A queued run's Budget starts at slot acquisition, not at run() entry.

    If Budget.started were captured before the semaphore, the second run
    would burn its 0.2s wall-clock limit purely from the 0.3s queue wait
    and terminate with BUDGET_TIME instead of DONE_UNVERIFIED.
    """
    llm = _GatedLLM()
    kwargs = {
        **service_kwargs,
        "llm": llm,
        "tools": ToolRegistry([], default_timeout_seconds=1.0, max_output_chars=1000),
        "budget_limits": BudgetLimits(
            max_steps=1, max_seconds=0.2, max_tokens=1000, max_cost_usd=0,
        ),
    }

    semaphore = asyncio.Semaphore(1)
    service = AgentService(**kwargs, semaphore=semaphore)

    # First run holds the slot while its LLM call is blocked.
    first = asyncio.create_task(service.run([_msg()]))
    await asyncio.wait_for(llm.entered.wait(), timeout=5.0)

    # Second run queues on the semaphore; the wait exceeds max_seconds.
    second = asyncio.create_task(service.run([_msg()]))

    await asyncio.sleep(0.3)
    llm.release.set()

    _, result = await asyncio.gather(first, second)

    assert result.reason == TerminationReason.DONE_UNVERIFIED
    assert result.answer == "ok"


async def test_semaphore_serializes_runs_over_time(service_kwargs, monkeypatch):
    """With Semaphore(1), run_graph intervals do not overlap.

    run() includes the semaphore wait, so its intervals always overlap for queued runs; run_graph runs entirely inside
    the critical section. delay=100ms: large enough to guarantee overlap if runs were concurrent.
    """
    import use_cases.agent_service as agent_service_mod

    llm = _DelayLLM(delay=0.1)
    kwargs = {
        **service_kwargs,
        "llm": llm,
        "tools": ToolRegistry([], default_timeout_seconds=1.0, max_output_chars=1000),
        "budget_limits": BudgetLimits(
            max_steps=1, max_seconds=2.0, max_tokens=1000, max_cost_usd=0,
        ),
    }
    semaphore = asyncio.Semaphore(1)
    service = AgentService(**kwargs, semaphore=semaphore)

    run_graph_intervals: list[tuple[float, float]] = []
    real_run_graph = agent_service_mod.run_graph

    async def timed_run_graph(*args, **kwargs):
        start = time.monotonic()
        try:
            return await real_run_graph(*args, **kwargs)
        finally:
            run_graph_intervals.append((start, time.monotonic()))

    monkeypatch.setattr(agent_service_mod, "run_graph", timed_run_graph)

    n = 3
    tasks = [asyncio.create_task(service.run([_msg()])) for _ in range(n)]
    await asyncio.gather(*tasks)

    assert len(run_graph_intervals) == n

    # Sort by start time: append order is by completion.
    run_graph_intervals.sort(key=lambda t: t[0])

    for (start_a, end_a), (start_b, _end_b) in zip(
            run_graph_intervals, run_graph_intervals[1:]
    ):
        assert end_a <= start_b, (
            f"runs overlapped: previous ended at {end_a}, next started at {start_b}"
        )


# ---------------------------------------------------------------------------
# agent_name: propagation into the Trace
# ---------------------------------------------------------------------------

async def test_agent_name_propagated_to_trace(service_kwargs, monkeypatch):
    """agent_name reaches Trace.agent_name for every run() call.

    The fake run_graph must close the trace in ``finally`` — otherwise the
    unclosed Trace emits a ResourceWarning at GC, which ``filterwarnings=error``
    turns into a test failure.
    """
    import use_cases.agent_service as agent_service_mod

    captured: dict = {}

    async def fake_run_graph(*args, **kwargs):
        trace = kwargs["trace"]
        captured["agent_name"] = trace.agent_name
        try:
            return _fake_result()
        finally:
            trace.close()

    monkeypatch.setattr(agent_service_mod, "run_graph", fake_run_graph)

    semaphore = asyncio.Semaphore(1)
    service = AgentService(
        **service_kwargs, semaphore=semaphore, agent_name="classifier_agent",
    )

    await service.run([_msg()])

    assert captured["agent_name"] == "classifier_agent"


async def test_agent_name_none_by_default(service_kwargs, monkeypatch):
    """Without agent_name in the constructor, Trace.agent_name stays None (backward compatible)."""
    import use_cases.agent_service as agent_service_mod

    captured: dict = {}

    async def fake_run_graph(*args, **kwargs):
        trace = kwargs["trace"]
        captured["agent_name"] = trace.agent_name
        try:
            return _fake_result()
        finally:
            trace.close()

    monkeypatch.setattr(agent_service_mod, "run_graph", fake_run_graph)

    semaphore = asyncio.Semaphore(1)
    service = AgentService(**service_kwargs, semaphore=semaphore)

    await service.run([_msg()])

    assert captured["agent_name"] is None
