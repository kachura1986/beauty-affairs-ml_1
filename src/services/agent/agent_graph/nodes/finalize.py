from typing import Any

from langgraph.runtime import Runtime

from ..services.finalize import FinalizeService
from ..state import AgentContext, GraphState
from ...domain import Finalization, TerminationReason


async def finalize_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Synthesize a final answer after max_steps has been exhausted.

    Preconditions (enforced here, not in FinalizeService):
        state['termination_reason'] == TerminationReason.BUDGET_STEPS.
        state['needs_finalization'] is True (set by decide_node only for BUDGET_STEPS).
        state['answer'] is None.

    This is the only place where an LLM call may run without can_start_step(); the exception is bounded to exactly
    one call and gated by policy.finalize_on_max_steps and hard-budget checks.

    Every return path resets state['needs_finalization'] to False, so the flag never leaks past this node
    into finish_node or subsequent debugging.

    All budget/LLM accounting and trace.error(kind='finalization_*') live in
    FinalizeService.run; this node only orchestrates.

    Args:
        state (GraphState): The current graph state.
        runtime (Runtime[AgentContext]): Runtime for the graph.

    Returns:
        dict[str, Any]: Result of the node.
    """
    if state.get('termination_reason') != TerminationReason.BUDGET_STEPS:
        raise RuntimeError(
            "finalize_node requires BUDGET_STEPS termination reason; "
            f"got {state.get('termination_reason')!r}."
        )

    if not state.get('needs_finalization'):
        # Defensive: routing sent us here by mistake.
        return {
            'finalization': Finalization.NONE,
            'needs_finalization': False,
        }

    ctx = runtime.context
    service = FinalizeService(llm=ctx.llm, trace=ctx.trace, task=ctx.task)

    answer, finalization = await service.run(
        budget=state['budget'],  # by reference — do not copy
        history=state['history'],
        last_tool_ok=state.get('last_tool_ok'),
    )

    return {
        'answer': answer,
        'finalization': finalization,
        'needs_finalization': False,
    }
