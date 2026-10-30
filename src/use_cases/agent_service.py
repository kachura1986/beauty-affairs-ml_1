from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

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
            policy: AgentPolicy | None = None,
            budget_limits: BudgetLimits | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._runs_dir = runs_dir
        self._policy = policy or AgentPolicy()
        self._budget_limits = budget_limits

    async def run(self, messages: Sequence[Message]) -> AgentResult:
        """
        Execute the agent_graph for the complete conversation history.

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

        history = [
            {
                'role': message.role,
                'content': message.content,
            }
            for message in messages
        ]

        if self._budget_limits is None:
            task = Task(system_prompt=self._system_prompt, initial_messages=history)
        else:
            task = Task(
                system_prompt=self._system_prompt,
                initial_messages=history,
                max_steps=self._budget_limits.max_steps,
                max_seconds=self._budget_limits.max_seconds,
                max_tokens=self._budget_limits.max_tokens,
                max_cost_usd=self._budget_limits.max_cost_usd,
            )

        trace = Trace(self._runs_dir)

        return await run_graph(
            task=task,
            tools=self._tools,
            llm=self._llm,
            trace=trace,
            policy=self._policy,
        )
