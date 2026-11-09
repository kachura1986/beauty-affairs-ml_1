import pytest

from services.agent.agent_graph.ports import Finalizer, StepRunner
from services.agent.agent_graph.services import FinalizeService, LLMStepService
from services.agent.agent_graph.state import AgentContext, create_initial_state
from services.agent.domain import AgentPolicy, BudgetLimits, Task
from services.agent.tools import Tool, ToolRegistry
from services.agent.trace import Trace

from _testing import EchoArgs, echo_fn, FakeLLM, reply


@pytest.fixture
def trace(tmp_path):
    t = Trace(tmp_path)
    yield t
    t.close()


@pytest.fixture
def make_registry():
    def _make(*extra_tools: Tool) -> ToolRegistry:
        tools = [
            Tool(name="echo", description="echo", fn=echo_fn, args_model=EchoArgs),
            *extra_tools,
        ]
        return ToolRegistry(tools, default_timeout_seconds=30.0, max_output_chars=15000)

    return _make


@pytest.fixture
def make_context(trace, make_registry):
    def _make(
            llm: FakeLLM | None = None,
            tools: ToolRegistry | None = None,
            postcondition=None,
            policy: AgentPolicy | None = None,
            max_steps: int = 16,
            max_seconds: float = 300.0,
            max_tokens: int = 200_000,
            max_cost_usd: float = 0.50,
            system_prompt: str = "sys",
            step_runner: StepRunner | None = None,
            finalizer: Finalizer | None = None,
    ) -> AgentContext:
        tools_ = tools or make_registry()
        llm_ = llm or FakeLLM(reply(text="ok"))

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
            tools=tools_,
            llm=llm_,
            trace=trace,
            policy=policy or AgentPolicy(empty_limit=3, repeat_limit=3),
            step_runner=step_runner or LLMStepService(llm=llm_, tools=tools_, trace=trace),
            finalizer=finalizer or FinalizeService(llm=llm_, trace=trace, task=task),
        )

    return _make


@pytest.fixture
def make_state():
    def _make(ctx: AgentContext):
        return create_initial_state(ctx.task)

    return _make


@pytest.fixture
def fast_retry(monkeypatch):
    """Shrink retry delays so tests do not sleep on real backoff.

    Both ``LLMStepService`` and ``FinalizeService`` go through ``_retry_backoff``
    and read settings on every call. Without this fixture, retry tests would add
    ~1.0s per test to the suite. The fixture lives in conftest so both
    ``test_decide.py`` and ``test_finalize.py`` see it without duplication.
    """
    from core.config import settings
    monkeypatch.setattr(settings, "LLM_RETRY_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr(settings, "LLM_RETRY_MIN_TIMEOUT_SECONDS", 0.01)
