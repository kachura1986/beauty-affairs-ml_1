import logging
import time
from typing import Any

from langgraph.runtime import Runtime

from core.config import settings
from services.llm import LLMModel, Reply
from .state import AgentContext, GraphState
from ..context import (
    AgentContextWindow,
    ContextOverflow,
    build_context,
    build_observation_message,
    build_tool_message,
)
from ..domain import AgentResult, Budget, TerminationReason
from ..tools import ToolRegistry, ToolStatus
from ..tools.errors import already_failed_result, repeated_action_result, skipped_due_to_termination_result
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
    Execute exactly one LLM decision step.

    Args:
        llm (LLMModel): LLM to use for the request.
        tools (ToolRegistry): Tool registry for the request.
        trace (Trace): Trace for the request.
        budget (Budget): Budget for the request.
        context_window (AgentContextWindow): Context for the request.

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

    # Pre-flight for tokens. Strict '<' on purpose: a caller that lands exactly
    # on MIN_EXPECTED_TOKENS is still allowed to try one more call.
    if budget.remaining_tokens() < settings.MIN_EXPECTED_TOKENS:
        logger.info(
            'Preflight token check failed | step=%s | remaining=%s | min=%s',
            budget.steps,
            budget.remaining_tokens(),
            settings.MIN_EXPECTED_TOKENS,
            extra={'step': budget.steps, 'error_code': 'preflight_tokens'},
        )
        trace.error(
            step=budget.steps,
            kind='preflight_tokens',
            message='budget_tokens',
            latency_ms=0,
            budget=budget,
        )

        return None, TerminationReason.BUDGET_TOKENS

    budget.steps += 1

    t0 = time.monotonic()

    try:
        chat = getattr(llm, 'chat', None)

        if chat is None:
            raise ValueError(f"{llm.__class__.__name__} does not support agent chat()")

        timeout = min(remaining, settings.LLM_DEFAULT_TIMEOUT_SECONDS)
        reply = await chat(context_window.messages, tools.schemas(), timeout, context_window.system)

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
    trace.step(
        budget.steps,
        context_window.messages,
        reply,
        budget,
        latency_ms,
        system_prompt=context_window.system,
    )

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
            'history': [build_observation_message('Empty response. Choose a tool or provide a final answer.')],
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

    # INVARIANT: len(history_updates) == len(state['pending_tool_calls']) on every exit path.
    # Never use `break` here — every call must receive a matching tool-message.
    for call in state['pending_tool_calls']:
        fp = ctx.tools.fingerprint(call)

        # Step 1. Run is already terminating: report this call and every
        # remaining one as skipped, so the assistant tool_calls stay paired.
        if termination_reason is not None:
            history_updates.append(build_tool_message(
                call,
                skipped_due_to_termination_result(call.get('name', '?'), termination_reason.value),
            ))
            continue

        # Step 2. This exact call already failed irrecoverably: return a
        # structured non-retryable error instead of an observation.
        if fp in failed_set:
            history_updates.append(build_tool_message(call, already_failed_result(call.get('name', '?'))))

            continue

        # Step 3. Count repeated identical actions; exceeding the limit
        # terminates the run but does not skip this call's response.
        count = action_counts.get(fp, 0) + 1
        action_counts[fp] = count

        if count > ctx.policy.repeat_limit:
            termination_reason = TerminationReason.REPEATED_ACTION
            history_updates.append(build_tool_message(call, repeated_action_result(call.get('name', '?'))))

            continue  # remaining calls are handled by branch (1)

        # Step 4. Pre-flight: budget may have expired since _request_llm.
        # `remaining` is computed once and reused for invoke() — no recomputation.
        remaining = budget.remaining_seconds()
        if remaining <= 0:
            termination_reason = TerminationReason.BUDGET_TIME
            history_updates.append(build_tool_message(
                call,
                skipped_due_to_termination_result(call.get('name', '?'), termination_reason.value),
            ))
            continue

        # Step 5. Execute the tool call with a timeout capped by both the remaining wall-clock budget and
        # the per-tool ceiling (the latter is applied inside invoke()).
        t0 = time.monotonic()
        res = await ctx.tools.invoke(call, timeout=remaining)
        latency_ms = int((time.monotonic() - t0) * 1000)
        ctx.trace.tool(budget.steps, call, res, latency_ms)

        # Remember irrecoverable failures so repeats are answered synthetically.
        if res.status is ToolStatus.ERROR and not res.retryable:
            failed_set.add(fp)
            if fp not in failed_actions:
                failed_actions.append(fp)

        history_updates.append(build_tool_message(call, res))

        # Step 6. Hard budget check after each call: record the reason but
        # keep iterating so every remaining call still gets a response.
        reason = budget.exceeded_without_steps()
        if reason:
            termination_reason = reason

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
            build_observation_message(f'Postcondition not satisfied: {note}')
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
        run_id=runtime.context.trace.run_id,
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
