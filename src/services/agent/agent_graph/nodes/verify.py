from typing import Any

from langgraph.runtime import Runtime

from ...context import build_observation_message
from ...domain import TerminationReason
from .._common import _log_internal_exception
from ..state import AgentContext, GraphState


def verify_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Equivalent of the existing VERIFY phase.

    Args:
        state: The current state of the agent_graph.
        runtime: The runtime environment of the graph.

    Returns:
        dict[str, Any]: The result of the verification.
    """
    ctx = runtime.context

    answer = state['pending_answer']

    if answer is None:
        raise RuntimeError('VERIFY phase has no pending answer.')

    # No postcondition: preserve the old DONE_UNVERIFIED behavior
    if ctx.task.postcondition is None:
        return {
            'answer': answer,
            'pending_answer': None,
            'termination_reason': (
                TerminationReason.DONE_UNVERIFIED
            ),
        }

    # Run configured postcondition
    try:
        ok, note = ctx.task.postcondition(answer)

    except Exception as exc:  # noqa: BLE001
        _log_internal_exception('Postcondition verification failed.', exc)
        ctx.trace.error(
            step=state['budget'].steps,
            kind='postcondition_error',
            message=type(exc).__name__,
            latency_ms=0,
            budget=state['budget'],
        )
        return {
            'answer': answer,
            'pending_answer': None,
            'termination_reason': TerminationReason.POSTCONDITION_ERROR,
        }

    # Verification succeeded
    if ok:
        return {
            'answer': answer,
            'pending_answer': None,
            'termination_reason': TerminationReason.DONE,
        }

    # Verification failed: go back to the model with an observation
    return {
        'pending_answer': None,
        'history': [
            build_observation_message(f'Postcondition not satisfied: {note}')
        ],
    }
