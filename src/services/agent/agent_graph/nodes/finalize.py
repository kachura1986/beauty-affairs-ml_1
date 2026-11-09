from typing import Any

from langgraph.runtime import Runtime

from ...domain import Finalization, TerminationReason
from ..state import AgentContext, GraphState


async def finalize_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Synthesize a final answer after max_steps has been exhausted.

    CONTRACT: caller guarantees termination_reason == BUDGET_STEPS and answer is None;
    needs_finalization is reset on every return path so it never leaks past this node.

    The only place where an LLM call may run without can_start_step(); bounded to exactly one call
    and gated by policy.finalize_on_max_steps and hard-budget checks. All budget/LLM accounting and
    trace.error(kind='finalization_*') live in FinalizeService.run; this node only orchestrates.
    The FinalizeService implementation is injected via ``AgentContext.finalizer``.

    Args:
    state (GraphState): The current graph state.
    runtime (Runtime[AgentContext]): Runtime for the graph.

    Returns:
        dict[str, Any]: Result of the node.
    """
    if state.get('termination_reason') != TerminationReason.BUDGET_STEPS:
        raise RuntimeError(
            f"finalize_node requires BUDGET_STEPS termination reason; got {state.get('termination_reason')!r}."
        )

    if not state.get('needs_finalization'):
        # Defensive: routing sent us here by mistake.
        return {
            'finalization': Finalization.NONE,
            'needs_finalization': False,
        }

    ctx = runtime.context

    answer, finalization = await ctx.finalizer.run(
        budget=state['budget'],  # by reference — do not copy
        history=state['history'],
        last_tool_ok=state.get('last_tool_ok'),
    )

    return {
        'answer': answer,
        'finalization': finalization,
        'needs_finalization': False,
    }
