import time
from typing import Any

from langgraph.runtime import Runtime

from core.config import settings
from ._common import FINALIZE_ANCHOR, _log_internal_exception
from ..state import AgentContext, GraphState
from ...domain import Finalization, TerminationReason


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
