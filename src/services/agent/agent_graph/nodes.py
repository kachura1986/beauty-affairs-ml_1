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
from ..domain import AgentResult, Budget, Finalization, TerminationReason
from ..tools import ToolRegistry, ToolStatus
from ..tools.errors import already_failed_result, repeated_action_result, skipped_due_to_termination_result
from ..trace import Trace

logger = logging.getLogger(__name__)

# Appended to the task system prompt for the single finalization LLM call that
# runs after max_steps has been exhausted. Tools are disabled for this call.
FINALIZE_ANCHOR = (
    'You have reached the final step. Provide the best final answer now using '
    'the information already collected. Do not call any tools.'
)


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


def _has_tool_results(history: list[dict[str, Any]]) -> bool:
    """
    Return True if history contains at least one tool message.

    Synthetic error messages (skipped_due_to_termination_result,
    already_failed_result, repeated_action_result) also count: the model can
    still synthesize a final answer that explains the failure. Excluding them
    would silently skip finalization in error scenarios, which is exactly the
    case the feature is meant to handle. An empty content field is also
    counted; _deterministic_fallback independently decides whether a usable
    result is present.

    Args:
        history (list[dict[str, Any]]): Conversation history.

    Returns:
        bool: True if at least one tool message is present.
    """
    return any(msg.get('role') == 'tool' for msg in history)


def _deterministic_fallback(history: list[dict[str, Any]], last_tool_ok: str | None) -> str | None:
    """
    Build a deterministic answer from the most recent successful tool result,
    or from the last tool error message when no successful result exists.

    A tool result is considered successful based on the explicit ``last_tool_ok`` field maintained by ``execute_node``;
    this removes the fragile ``startswith('ERROR ')`` heuristic over message content.

    When no successful tool result is available, the last tool message in history (which, by elimination,
    must be an error: synthetic errors from skipped_due_to_termination_result / already_failed_result /
    repeated_action_result, or a real execution error) is used as a fallback explanation, so the caller returns
    a non-empty answer that describes the failure instead of a silent None.

    Returns None when no tool message is present at all, so the caller can leave answer=None and finalization=NONE.

    Args:
        history (list[dict[str, Any]]): Conversation history.
        last_tool_ok (str | None): Content of the most recent successful tool result, or None if no tool call has succeeded.

    Returns:
        str | None: A deterministic answer, or None if no tool message is present.
    """
    if last_tool_ok is not None:
        return (
            'The model could not produce a final answer within the step limit. Here is the most recent tool result:\n'
            f'{last_tool_ok}'
        )

    # No successful tool result. Use the last tool message as an explanation;
    # by elimination it must be an error (real or synthetic).
    for msg in reversed(history):
        if msg.get('role') != 'tool':
            continue

        content = msg.get('content') or ''

        return (
            'The model could not produce a final answer within the step limit. The last tool call failed:\n'
            f'{content}'
        )

    return None


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
        trace.error(
            step=budget.steps,
            kind='preflight_tokens',
            message=f'remaining={budget.remaining_tokens()}, min={settings.MIN_EXPECTED_TOKENS}',
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
            extra={'step': budget.steps, 'context_overflow_code': exc.code},
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


async def finalize_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Synthesize a final answer after max_steps has been exhausted.

    Preconditions:
        state['termination_reason'] == TerminationReason.BUDGET_STEPS.
        state['needs_finalization'] is True (set by decide_node only for BUDGET_STEPS).
        state['answer'] is None.

    This is the only place where an LLM call may run without can_start_step(); the exception is bounded to exactly
    one call and gated by policy.finalize_on_max_steps and hard-budget checks.

    Every return path resets state['needs_finalization'] to False, so the flag never leaks past this node
    into finish_node or subsequent debugging.

    All fallback paths record ``trace.error(kind='finalization_fallback', ...)`` with a machine-readable message code,
    so post-mortem can distinguish the cause (budget exhaustion, LLM error, empty reply, tool calls in reply).
    A successful finalization still records ``trace.error(kind='finalization_overrun', ...)`` if the call pushed the
    run over a hard non-step budget; the termination reason itself is not changed (BUDGET_STEPS remains the primary reason).

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
    budget = state['budget']
    history = state['history']
    trace = ctx.trace
    last_tool_ok = state.get('last_tool_ok')

    # Hard budget pre-flight: the finalization call must not breach the remaining wall-clock or token budgets.
    if budget.remaining_seconds() <= 0 or budget.remaining_tokens() < settings.MIN_EXPECTED_TOKENS:
        trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message='budget_exhausted',
            latency_ms=0,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return {
            'answer': answer,
            'finalization': Finalization.FALLBACK if answer is not None else Finalization.NONE,
            'needs_finalization': False,
        }

    system_prompt = ctx.task.system_prompt + '\n\n' + FINALIZE_ANCHOR

    # Re-check remaining time to defend against a TOCTOU race between the budget check above and the actual call.
    remaining = budget.remaining_seconds()

    if remaining <= 0:
        trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message='budget_exhausted',
            latency_ms=0,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return {
            'answer': answer,
            'finalization': Finalization.FALLBACK if answer is not None else Finalization.NONE,
            'needs_finalization': False,
        }

    timeout = min(remaining, settings.LLM_DEFAULT_TIMEOUT_SECONDS)

    if timeout <= 0:
        trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message='budget_exhausted',
            latency_ms=0,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return {
            'answer': answer,
            'finalization': Finalization.FALLBACK if answer is not None else Finalization.NONE,
            'needs_finalization': False,
        }

    t0 = time.monotonic()

    try:
        chat = getattr(ctx.llm, 'chat', None)

        if chat is None:
            raise ValueError(f"{ctx.llm.__class__.__name__} does not support agent chat()")

        reply = await chat(history, [], timeout, system_prompt)
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Same helper used by _request_llm so exceptions are not silently swallowed by the fallback path.
        # step=budget.steps keeps the association with the last executed step.
        _log_internal_exception('Finalization LLM call failed.', exc, step=budget.steps)

        trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message='llm_error',
            latency_ms=latency_ms,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return {
            'answer': answer,
            'finalization': Finalization.FALLBACK if answer is not None else Finalization.NONE,
            'needs_finalization': False,
        }

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Account for the call regardless of outcome: tokens and cost were consumed.
    if reply is not None:
        budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost_usd)

    # reply.tool_calls is ignored on finalization: the model was given an empty tool list, but the Responses API
    # does not enforce it. We do not propagate them into pending_tool_calls, to keep the pairing invariant intact.
    if reply is None or not reply.text.strip():
        trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message='empty_reply',
            latency_ms=latency_ms,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return {
            'answer': answer,
            'finalization': Finalization.FALLBACK if answer is not None else Finalization.NONE,
            'needs_finalization': False,
        }

    if reply.tool_calls:
        trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message='tool_calls_in_reply',
            latency_ms=latency_ms,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return {
            'answer': answer,
            'finalization': Finalization.FALLBACK if answer is not None else Finalization.NONE,
            'needs_finalization': False,
        }

    # A successful finalization may itself push us over a hard budget. Record this for observability without
    # changing the termination reason: BUDGET_STEPS remains the primary reason, and the budget snapshot
    # in trace.finish carries the actual usage.
    overrun = budget.exceeded_without_steps()

    if overrun is not None:
        trace.error(
            step=budget.steps,
            kind='finalization_overrun',
            message=overrun.value,
            latency_ms=latency_ms,
            budget=budget,
        )

    return {
        'answer': reply.text,
        'finalization': Finalization.LLM,
        'needs_finalization': False,
    }


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

    # Preserve the most recent successful tool result across execute rounds;
    # only overwrite it when a tool in this round returns ToolStatus.OK.
    last_tool_ok = state.get('last_tool_ok')

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

        # Track the most recent successful tool result explicitly, so the
        # finalization fallback does not have to guess from message content.
        if res.status is ToolStatus.OK:
            last_tool_ok = res.content

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
        'last_tool_ok': last_tool_ok,
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
