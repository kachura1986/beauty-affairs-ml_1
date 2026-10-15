from langgraph.graph import END, START, StateGraph

from .nodes import decide_node, execute_node, finish_node, verify_node
from .routing import route_after_decide, route_after_execute, route_after_verify
from .state import AgentContext, GraphState


def build_graph():
    """
    Build and compile the LangGraph workflow.

    Returns:
        Compiled graph.
    """
    builder = StateGraph(state_schema=GraphState, context_schema=AgentContext)

    # Nodes
    builder.add_node('decide', decide_node)
    builder.add_node('execute', execute_node)
    builder.add_node('verify', verify_node)
    builder.add_node('finish', finish_node)

    # Edges
    # Entry
    builder.add_edge(START, 'decide')

    # DECIDE -> EXECUTE / VERIFY / FINISH / DECIDE
    builder.add_conditional_edges('decide', route_after_decide)

    # EXECUTE -> DECIDE / FINISH
    builder.add_conditional_edges('execute', route_after_execute)

    # VERIFY -> DECIDE / FINISH
    builder.add_conditional_edges('verify', route_after_verify)

    # FINISH -> END
    builder.add_edge('finish', END)

    return builder.compile()


# Compiled graph can be reused between runs because all per-run dependencies are supplied via runtime context
GRAPH = build_graph()
