from typing import Literal

from .state import GraphState


def route_after_decide(state: GraphState) -> Literal['execute', 'verify', 'finish', 'decide']:
    """
    Decide where the graph goes after an LLM decision.
    """
    if state.get('termination_reason') is not None:
        return 'finish'

    if state.get('pending_tool_calls'):
        return 'execute'

    if state.get('pending_answer') is not None:
        return 'verify'

    # Empty response that hasn't hit the limit yet
    return 'decide'


def route_after_execute(state: GraphState) -> Literal['finish', 'decide']:
    """
    After tool execution.
        
    Args:
        state: Current state of the graph.
        
    Returns:
        "finish" -> Terminal state
        "decide" -> Continue decision
    """
    if state.get('termination_reason') is not None:
        return 'finish'

    return 'decide'


def route_after_verify(state: GraphState) -> Literal['finish', 'decide']:
    """
    After postcondition verification
    
    Args:
        state: Current state of the graph.
        
    Returns:
        "finish" -> Terminal state
        "decide" -> Continue decision
    """
    if state.get('termination_reason') is not None:
        return 'finish'

    return 'decide'
