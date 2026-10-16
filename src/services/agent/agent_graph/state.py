import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from services.llm import LLMModel
from ..domain import AgentPolicy, AgentResult, Budget, BudgetLimits, Task, TerminationReason
from ..tools import ToolRegistry
from ..trace import Trace


@dataclass(frozen=True)
class AgentContext:
    """
    Runtime dependencies for one agent_graph execution.

    These are deliberately kept outside GraphState because they are
    dependencies of the workflow rather than workflow state.
    """

    task: Task
    tools: ToolRegistry
    llm: LLMModel
    trace: Trace
    policy: AgentPolicy


class GraphState(TypedDict, total=False):
    """
    Mutable workflow state managed by LangGraph.

    This is intentionally close to the existing AgentState, with a few
    extra fields used for communication between graph nodes.
    """

    # Conversation history is append-only
    history: Annotated[list[dict[str, Any]], operator.add]

    # Existing runtime resource accounting
    budget: Budget

    # Existing anti-repeat / anti-loop state
    action_counts: dict[str, int]
    failed_actions: list[str]
    consecutive_empty_steps: int
    budget_warning_emitted: bool

    # Graph-local pending values
    pending_tool_calls: list[dict[str, Any]]
    pending_answer: str | None

    # Terminal state
    termination_reason: TerminationReason | None
    answer: str | None
    result: AgentResult


def create_initial_state(task: Task) -> GraphState:
    """
    Create the initial graph state.

    Args:
        task (Task): The task to run.

    Returns:
        GraphState: The initial graph state.
    """
    history = task.messages or [
        {
            'role': 'user',
            'content': task.prompt,
        },
    ]

    return {
        # Conversation history is append-only
        'history': history,

        # Existing runtime resource accounting
        'budget': Budget(
            limits=BudgetLimits(
                max_steps=task.max_steps,
                max_seconds=task.max_seconds,
                max_tokens=task.max_tokens,
                max_cost_usd=task.max_cost_usd,
            )
        ),

        # Existing anti-repeat / anti-loop state
        'action_counts': {},
        'failed_actions': [],
        'consecutive_empty_steps': 0,
        'budget_warning_emitted': False,

        # Graph-local pending values
        'pending_tool_calls': [],
        'pending_answer': None,

        # Terminal state
        'termination_reason': None,
        'answer': None,
    }
