from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol
from asyncio import Semaphore

from services.agent.agent_graph import run_graph
from services.agent.domain import AgentPolicy, AgentResult, BudgetLimits, Task
from services.agent.tools import ToolRegistry
from services.agent.trace import Trace
from services.llm import LLMModel


class Message(Protocol):
    """
    Minimal structural contract for a client chat message accepted by AgentService.

    Only ``user`` and ``assistant`` roles are allowed. System prompts are supplied by the service itself
    (``Task.system_prompt``), and tool messages are produced exclusively by the internal agent loop, never by the client.
    """

    role: Literal['user', 'assistant']
    content: str


class AgentService:
    """The AgentService is a high-level interface for running agent_graphs."""

    def __init__(
            self,
            llm: LLMModel,
            tools: ToolRegistry,
            system_prompt: str,
            runs_dir: Path,
            policy: AgentPolicy,
            budget_limits: BudgetLimits,
            semaphore: Semaphore,
    ) -> None:
        """
        Initialize the service.

        CONTRACT: ``semaphore`` is injected by the composition root (PipelineDependencies) so the concurrency limit is
        configured in one place and can later be shared with other services hitting the same upstream.

        Args:
            llm (LLMModel): LLM used for decision and finalization steps.
            tools (ToolRegistry): Tool registry available to every run.
            system_prompt (str): System instructions used to build the context window.
            runs_dir (Path): Directory for trace JSONL files.
            policy (AgentPolicy): Anti-loop / finalization policy applied per run.
            budget_limits (BudgetLimits): Per-run resource limits.
            semaphore (Semaphore): Shared concurrency guard for ``run()``.
        """
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._runs_dir = runs_dir
        self._policy = policy
        self._budget_limits = budget_limits
        self._semaphore = semaphore

    async def run(self, messages: Sequence[Message]) -> AgentResult:
        """
        Execute the agent_graph for the complete conversation history.

        CONTRACT: blocks on the injected semaphore until a slot is free. Validation errors are raised before
        acquisition (fail fast, no slot wasted). ``Budget.max_seconds`` starts only inside the slot, so queue
        wait does not count against a run's wall-clock budget.

        Args:
            messages (Sequence[Message]): Complete conversation history.

        Returns:
            AgentResult: Stable result of the agent_graph execution.
        """
        if not messages:
            raise ValueError(f"{self.__class__.__name__}: Messages cannot be empty.")

        for message in messages:
            if message.role not in ('user', 'assistant'):
                raise ValueError(
                    f"{self.__class__.__name__}: unsupported role {message.role!r}; "
                    "only 'user' and 'assistant' are allowed."
                )

        async with self._semaphore:
            history = [
                {
                    'role': message.role,
                    'content': message.content,
                }
                for message in messages
            ]

            task = Task(
                system_prompt=self._system_prompt,
                initial_messages=history,
                budget_limits=self._budget_limits,
            )

            trace = Trace(self._runs_dir)

            return await run_graph(
                task=task,
                tools=self._tools,
                llm=self._llm,
                trace=trace,
                policy=self._policy,
            )
