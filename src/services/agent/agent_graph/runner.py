from services.llm import LLMModel
from .graph import GRAPH
from .state import AgentContext, create_initial_state
from ..domain import AgentPolicy, AgentResult, Task
from ..tools import ToolRegistry
from ..trace import Trace


async def run_graph(
        task: Task,
        tools: ToolRegistry,
        llm: LLMModel,
        trace: Trace,
        policy: AgentPolicy,
) -> AgentResult:
    """
    Run the graph.

    Args:
        task (Task): The task to run.
        tools (ToolRegistry): The tools to use.
        llm (LLMModel): The LLM to use.
        trace (Trace): The trace to use.
        policy (AgentPolicy): The policy to use.
    """
    state = await GRAPH.ainvoke(
        create_initial_state(task),
        context=AgentContext(
            task=task,
            tools=tools,
            llm=llm,
            trace=trace,
            policy=policy,
        ),
        config={
            # Your max_steps counts LLM calls. LangGraph's recursion_limit counts graph super-steps,
            # so they are intentionally not treated as the same value
            'recursion_limit': max(64, task.max_steps * 4 + 16),
        },
    )

    result = state.get('result')

    if result is None:
        raise RuntimeError("Graph completed without an AgentResult.")

    return result
