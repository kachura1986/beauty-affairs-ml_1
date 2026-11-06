"""Tests for AgentService concurrency limiting."""
import asyncio
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

    async def fake_run_graph(*args, **kwargs):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
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

    # Let all four attempt acquisition; only 2 must reach run_graph.
    await asyncio.sleep(0.05)
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

    with pytest.raises(ValueError):
        await service.run([])
    with pytest.raises(ValueError):
        await service.run([_msg(role="system", content="bad")])

    # _value is private, but in a test it proves the slot was never acquired.
    assert semaphore._value == 1

    await service.run([_msg()])
    assert len(calls) == 1