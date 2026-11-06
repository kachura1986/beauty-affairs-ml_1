from typing import Any

from langgraph.runtime import Runtime

from ...domain import AgentResult, Finalization
from ..state import AgentContext, GraphState


def finish_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Finish the agent_graph execution and build the stable AgentResult.

    CONTRACT: requires a termination_reason; raising here surfaces a routing bug
    rather than emitting a result with an unknown reason.

    Args:
        state (GraphState): The current state of the agent_graph execution.
        runtime (Runtime[AgentContext]): The runtime environment of the graph.

    Returns:
        dict[str, Any]: The result of the node.
    """
    reason = state.get('termination_reason')
    if reason is None:
        raise RuntimeError('Finish node requires a termination reason.')

    answer = state.get('answer')
    finalization = state.get('finalization', Finalization.NONE)
    runtime.context.trace.finish(reason, state['budget'], answer, finalization=finalization)

    result = AgentResult(
        run_id=runtime.context.trace.run_id,
        reason=reason,
        answer=answer,
        steps=state['budget'].steps,
        tokens=state['budget'].tokens,
        cost_usd=round(state['budget'].cost_usd, 6),
        seconds=round(state['budget'].elapsed(), 2),
        history_len=len(state['history']),
        trace=str(runtime.context.trace.path),
        finalization=finalization,
    )

    return {
        'result': result,
    }
