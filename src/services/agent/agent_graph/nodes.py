import logging
import time
from typing import Any

from langgraph.runtime import Runtime

from services.llm import LLMModel, Reply
from .state import AgentContext, GraphState
from ..context import build_window, observation, tool_message
from ..domain import AgentResult, TerminationReason
from ..tools import ToolRegistry, ToolStatus
from ..trace import Trace

logger = logging.getLogger(__name__)


def _log_internal_exception(message: str, exc: Exception, step: int | None = None) -> None:
    extra = {'step': step}

    logger.error(
        '%s | step=%s',
        message,
        step,
        exc_info=(type(exc), exc, exc.__traceback__),
        extra=extra,
    )


def _check_budget(budget) -> TerminationReason | None:
    """
    Preserve the existing agent_graph budget semantics.

    Step budget is checked first; then time/token/cost limits.
    """
    if not budget.can_start_step():
        return TerminationReason.BUDGET_STEPS

    return budget.exceeded_without_steps()


async def _request_llm(
        *,
        llm: LLMModel,
        tools: ToolRegistry,
        trace: Trace,
        budget,
        window: list[dict[str, Any]],
) -> tuple[Reply | None, TerminationReason | None]:
    """
    Execute exactly one LLM decision step.

    Args:
        llm (LLMProvider): LLM to use for the request.
        tools (ToolRegistry): Tool registry for the request.
        trace (Trace): Trace for the request.
        budget (Budget): Budget for the request.
        window (list[dict[str, Any]]): Context window for the request.

    Returns:
        tuple[Reply | None, TerminationReason | None]: Tuple of (reply, termination reason).
    """
    # Do not start an LLM step when the step budget is exhausted
    if not budget.can_start_step():
        return None, TerminationReason.BUDGET_STEPS

    # Limit the request by the remaining wall-clock budget
    remaining = budget.remaining_seconds()

    if remaining <= 0:
        return None, TerminationReason.BUDGET_TIME

    budget.steps += 1

    t0 = time.monotonic()

    try:
        chat = getattr(llm, 'chat', None)

        if chat is None:
            raise ValueError(f"{llm.__class__.__name__} does not support agent chat()")

        reply = await chat(window, tools.schemas(), remaining)

    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.monotonic() - t0) * 1000)

        _log_internal_exception("LLM request failed.", exc, step=budget.steps)

        if budget.remaining_seconds() <= 0:
            trace.error(
                step=budget.steps,
                kind='llm_timeout',
                message=type(exc).__name__,
                latency_ms=latency_ms,
                budget=budget,
            )

            return None, TerminationReason.BUDGET_TIME

        trace.error(
            step=budget.steps,
            kind='llm_error',
            message=type(exc).__name__,
            latency_ms=latency_ms,
            budget=budget,
        )

        return None, TerminationReason.LLM_ERROR

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Account for model usage
    budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost_usd)

    # Preserve existing trace format
    trace.step(budget.steps, window, reply, budget, latency_ms)

    # A completed LLM call may itself push us over a hard budget
    reason = budget.exceeded_without_steps()

    if reason:
        return None, reason

    return reply, None


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

    # Same context window construction as before
    window = build_window(ctx.task, state['history'], warning)

    reply, reason = await _request_llm(
        llm=ctx.llm,
        tools=ctx.tools,
        trace=ctx.trace,
        budget=budget,
        window=window,
    )

    if reason:
        return {
            'budget': budget,
            'termination_reason': reason,
            'budget_warning_emitted': budget_warning_emitted,
        }

    if reply is None:
        raise RuntimeError('LLM request returned no reply and no termination reason.')

    # ------------------------------------------------------------
    # Empty response
    # ------------------------------------------------------------
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
            'history': [
                observation('Empty response. Choose a tool or provide a final answer.')
            ],
        }

    # Any non-empty response resets the empty-response counter
    result: dict[str, Any] = {
        'budget': budget,
        'consecutive_empty_steps': 0,
        'budget_warning_emitted': budget_warning_emitted,
    }

    # ------------------------------------------------------------
    # Final answer candidate
    # ------------------------------------------------------------
    if not reply.tool_calls:
        result['pending_answer'] = reply.text
        return result

    # ------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------
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


async def execute_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Equivalent of Agent.execute_tools().

    Tool calls are still executed sequentially, exactly as in the original implementation.

    Args:
        state (GraphState): The current graph state.
        runtime (Runtime[AgentContext]): The runtime environment of the graph.

    Returns:
        dict[str, Any]: Result of the node.
    """
    ctx = runtime.context

    budget = state['budget']
    action_counts = dict(state['action_counts'])
    failed_actions = list(state['failed_actions'])
    failed_set = set(failed_actions)
    history_updates: list[dict[str, Any]] = []
    termination_reason: TerminationReason | None = None

    for call in state['pending_tool_calls']:
        fp = ctx.tools.fingerprint(call)

        # --------------------------------------------------------
        # Irrecoverably failed action
        # --------------------------------------------------------
        #
        # This intentionally does NOT terminate the whole execution
        # This matches the old _execute_tool_call():
        #
        #   observation(...)
        #   return None
        #
        # execute_tools() then proceeds to the next call
        if fp in failed_set:
            history_updates.append(
                observation('This tool call has already failed irrecoverably. Choose a different approach.')
            )
            continue

        # --------------------------------------------------------
        # Count repeated identical actions
        # --------------------------------------------------------
        count = action_counts.get(fp, 0) + 1
        action_counts[fp] = count

        if count > ctx.policy.repeat_limit:
            termination_reason = TerminationReason.REPEATED_ACTION
            break

        # --------------------------------------------------------
        # Execute tool
        # --------------------------------------------------------
        t0 = time.monotonic()

        res = await ctx.tools.invoke(call)

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Preserve existing trace
        ctx.trace.tool(budget.steps, call, res, latency_ms)

        # --------------------------------------------------------
        # Remember irrecoverable failures
        # --------------------------------------------------------
        if res.status is ToolStatus.ERROR and not res.retryable:
            failed_set.add(fp)

            if fp not in failed_actions:
                failed_actions.append(fp)

        # --------------------------------------------------------
        # Add result to conversation history
        # --------------------------------------------------------
        history_updates.append(tool_message(call, res))

        # --------------------------------------------------------
        # Check hard budget after every tool call
        # --------------------------------------------------------
        termination_reason = (budget.exceeded_without_steps())

        if termination_reason:
            break

    result: dict[str, Any] = {
        'history': history_updates,
        'action_counts': action_counts,
        'failed_actions': failed_actions,
        'pending_tool_calls': [],
        'budget': budget,
    }

    if termination_reason:
        result['termination_reason'] = termination_reason

    return result


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
        ok = False
        note = 'Postcondition verification failed due to an internal error.'

    # ------------------------------------------------------------
    # Verification succeeded
    # ------------------------------------------------------------
    if ok:
        return {
            'answer': answer,
            'pending_answer': None,
            'termination_reason': TerminationReason.DONE,
        }

    # ------------------------------------------------------------
    # Verification failed: go back to the model with an observation
    # ------------------------------------------------------------
    return {
        'pending_answer': None,
        'history': [
            observation(f'Postcondition not satisfied: {note}')
        ],
    }


def finish_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Finish the agent_graph execution.

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
    runtime.context.trace.finish(reason, state['budget'], answer)

    result = AgentResult(
        reason=reason,
        answer=answer,
        steps=state['budget'].steps,
        tokens=state['budget'].tokens,
        cost_usd=round(state['budget'].cost_usd, 6),
        seconds=round(state['budget'].elapsed(), 2),
        history_len=len(state['history']),
        trace=str(runtime.context.trace.path),
    )

    return {
        'result': result,
    }
