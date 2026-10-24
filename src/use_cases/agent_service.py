from collections.abc import Sequence
from uuid import uuid4

from services.agent.agent_graph import run_graph
from services.agent.domain import AgentPolicy, AgentResult, Task
from services.agent.tools import ToolRegistry
from services.agent.trace import Trace
from services.llm import LLMModel, Message


class AgentService:
    """Service that executes the agent_graph without HTTP."""

    def __init__(
            self,
            llm: LLMModel,
            tools: ToolRegistry,
            policy: AgentPolicy | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._policy = policy or AgentPolicy()

    async def run(self, messages: Sequence[Message]) -> AgentResult:
        """
        Execute the agent_graph for the complete conversation history.

        Args:
            messages (Sequence[Message]): Complete conversation history.

        Returns:
            AgentResult: Stable result of the agent_graph execution.
        """
        if not messages:
            raise ValueError('Messages cannot be empty.')

        history = [
            {
                'role': message.role,
                'content': message.content,
            }
            for message in messages
        ]

        prompt = next(
            (
                message['content']
                for message in reversed(history)
                if message['role'] == 'user'
            ),
            '',
        )

        if not prompt:
            raise ValueError('Messages must contain at least one user message.')

        task = Task(prompt=prompt, messages=history)

        trace = Trace(str(uuid4()))

        return await run_graph(
            task=task,
            tools=self._tools,
            llm=self._llm,
            trace=trace,
            policy=self._policy,
        )
