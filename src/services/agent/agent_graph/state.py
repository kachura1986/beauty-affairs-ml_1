import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from services.llm import LLMModel, ToolCall

from ..domain import AgentPolicy, AgentResult, Budget, Finalization, Task, TerminationReason
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

    # Content of the most recent successful tool result; set by execute_node when a ToolResult has status ToolStatus.OK.
    # Used by finalize_node as a deterministic fallback source, so the fallback does not rely on a
    # textual heuristic over tool message content.
    last_tool_ok: str | None

    # Graph-local pending values
    pending_tool_calls: list[ToolCall]
    pending_answer: str | None

    # Terminal state
    termination_reason: TerminationReason | None
    answer: str | None
    result: AgentResult

    # Finalization after max_steps: needs_finalization is set by decide_node only for BUDGET_STEPS, consumed and
    # reset by finalize_node. finalization is set by finalize_node and read by finish_node.
    needs_finalization: bool
    finalization: Finalization


def create_initial_state(task: Task) -> GraphState:
    """
    Create the initial graph state.

    Args:
        task (Task): The task to run.

    Returns:
        GraphState: The initial graph state.
    """
    history = list(task.initial_messages)

    return {
        # Conversation history is append-only
        'history': history,

        # Existing runtime resource accounting
        'budget': Budget(limits=task.budget_limits),

        # Existing anti-repeat / anti-loop state
        'action_counts': {},
        'failed_actions': [],
        'consecutive_empty_steps': 0,
        'budget_warning_emitted': False,

        # Most recent successful tool result
        'last_tool_ok': None,

        # Graph-local pending values
        'pending_tool_calls': [],
        'pending_answer': None,

        # Terminal state
        'termination_reason': None,
        'answer': None,

        # Finalization after max_steps
        'needs_finalization': False,
        'finalization': Finalization.NONE,
    }
