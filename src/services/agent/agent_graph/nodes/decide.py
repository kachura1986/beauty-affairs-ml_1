import logging
from typing import Any

from langgraph.runtime import Runtime

from services.llm import LLMModel, Reply

from ...context import AgentContextWindow, ContextOverflow, build_context, build_observation_message
from ...domain import Budget, TerminationReason
from ...tools import ToolRegistry
from ...trace import Trace
from .._fallback import _has_tool_results
from ..services.llm_step import LLMStepService
from ..state import AgentContext, GraphState

logger = logging.getLogger(__name__)


def _check_budget(budget: Budget) -> TerminationReason | None:
    """
    Preserve the existing agent_graph budget semantics.
    Step budget is checked first; then time/token/cost limits.

    Args:
        budget (Budget): Budget to check.

    Returns:
        TerminationReason | None: Termination reason if the budget is exceeded, otherwise None.
    """
    if not budget.can_start_step():
        return TerminationReason.BUDGET_STEPS

    return budget.exceeded_without_steps()


async def _request_llm(
        *,
        llm: LLMModel,
        tools: ToolRegistry,
        trace: Trace,
        budget: Budget,
        context_window: AgentContextWindow,
) -> tuple[Reply | None, TerminationReason | None]:
    """
    Thin compatibility wrapper around :class:`LLMStepService`.

    Preserved because existing tests monkeypatch ``nodes._request_llm``
    (e.g. ``test_decide_forwards_context_window_to_request_llm``). New code should call ``LLMStepService`` directly.

    Args:
        llm (LLMModel): LLM to use for the request.
        tools (ToolRegistry): Tool registry for the request.
        trace (Trace): Trace for the request.
        budget (Budget): Budget for the request (mutated by reference).
        context_window (AgentContextWindow): Context for the request.

    Returns:
        tuple[Reply | None, TerminationReason | None]: Tuple of (reply, termination reason).
    """
    return await LLMStepService(llm=llm, tools=tools, trace=trace).run(budget=budget, context_window=context_window)


async def decide_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Equivalent of:

        OBSERVE -> DECIDE -> handle_reply()

    The OBSERVE phase is collapsed into this node because it only built the context window.

    Args:
        state (GraphState): The current graph state.
        runtime (Runtime[AgentContext]): Runtime for the graph.

    Returns:
        dict[str, Any]: Result of the node.
    """
    ctx = runtime.context
    budget = state['budget']

    # Same budget check that previously happened before OBSERVE
    reason = _check_budget(budget)

    if reason:
        # BUDGET_STEPS with policy enabled and tool results present: hand control to finalize_node so the model can
        # synthesize an answer from history instead of silently finishing with answer=None.
        if (
                reason == TerminationReason.BUDGET_STEPS
                and ctx.policy.finalize_on_max_steps
                and _has_tool_results(state['history'])
        ):
            return {
                'budget': budget,
                'termination_reason': reason,
                'needs_finalization': True,
            }

        return {
            'termination_reason': reason,
            'budget': budget,
        }

    # Build the optional soft budget warning
    warning = None
    budget_warning_emitted = state['budget_warning_emitted']

    if not budget_warning_emitted:
        warning = budget.soft_warning()

        if warning:
            budget_warning_emitted = True

    # System prompt goes via its own channel; `messages` holds only history.
    # `ContextOverflow` → normal TerminationReason, finished via `finish_node`.
    try:
        context_window = build_context(ctx.task.system_prompt, warning, state['history'])
    except ContextOverflow as exc:
        # trace.error records exc.code; str(exc) goes to the logger (diagnostics only).
        # Alert on trace.error(kind='context_overflow'), not on this log line.
        logger.info(
            'Context overflow: %s (step=%s): %s',
            exc.code,
            budget.steps,
            str(exc),
            extra={'step': budget.steps},
        )

        ctx.trace.error(
            step=budget.steps,
            kind='context_overflow',
            message=exc.code,
            latency_ms=0,
            budget=budget,
        )

        return {
            'budget': budget,
            'budget_warning_emitted': budget_warning_emitted,
            'termination_reason': TerminationReason.CONTEXT_OVERFLOW,
        }

    reply, reason = await _request_llm(
        llm=ctx.llm,
        tools=ctx.tools,
        trace=ctx.trace,
        budget=budget,
        context_window=context_window,
    )

    if reason:
        result: dict[str, Any] = {
            'budget': budget,
            'termination_reason': reason,
            'budget_warning_emitted': budget_warning_emitted,
        }

        # Keep the final answer from the last LLM call before the budget was exceeded.
        # Tool calls don't qualify; postcondition is skipped (run ended on budget, not DONE).
        if reply is not None and not reply.tool_calls and reply.text.strip():
            result['answer'] = reply.text

        return result

    if reply is None:
        raise RuntimeError('LLM request returned no reply and no termination reason.')

    # Empty response
    if not reply.tool_calls and not reply.text.strip():
        consecutive_empty_steps = (state['consecutive_empty_steps'] + 1)

        if consecutive_empty_steps > ctx.policy.empty_limit:
            return {
                'budget': budget,
                'consecutive_empty_steps': consecutive_empty_steps,
                'budget_warning_emitted': budget_warning_emitted,
                'termination_reason': TerminationReason.NO_PROGRESS,
            }

        return {
            'budget': budget,
            'consecutive_empty_steps': consecutive_empty_steps,
            'budget_warning_emitted': budget_warning_emitted,
            'history': [build_observation_message('Empty response. Choose a tool or provide a final answer.')],
        }

    # Any non-empty response resets the empty-response counter
    result: dict[str, Any] = {
        'budget': budget,
        'consecutive_empty_steps': 0,
        'budget_warning_emitted': budget_warning_emitted,
    }

    # Final answer candidate
    if not reply.tool_calls:
        result['pending_answer'] = reply.text

        return result

    # Tool calls
    # Preserve the assistant message containing the tool calls
    result['history'] = [
        {
            'role': 'assistant',
            'content': reply.text or None,
            'tool_calls': reply.tool_calls,
        }
    ]

    result['pending_tool_calls'] = reply.tool_calls

    # Explicitly clear a possible stale answer
    result['pending_answer'] = None

    return result
