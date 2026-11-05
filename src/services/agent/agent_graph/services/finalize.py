import time
from typing import Any

from core.config import settings
from services.llm import LLMModel

from ...domain import Budget, Finalization, Task
from ...trace import Trace
from .._common import FINALIZE_ANCHOR, _log_internal_exception
from .._fallback import _deterministic_fallback


class FinalizeService:
    """
    The single bounded finalization LLM call after max_steps is exhausted.

    Preconditions (enforced by ``finalize_node``, not by this service):
        ``termination_reason == BUDGET_STEPS``, ``needs_finalization is True``,
        ``answer is None``, ``policy.finalize_on_max_steps is True``.

    Returns ``(answer, finalization)``. Does NOT return ``termination_reason`` and
    does NOT write it to state: an overrun is recorded only as
    ``trace.error(kind='finalization_overrun')``; the reason remains ``BUDGET_STEPS``
    as set by ``decide_node``.

    Does NOT write ``needs_finalization`` — that is the node's contract.

    All hard-budget pre-flight and fallback paths write
    ``trace.error(kind='finalization_fallback', message=<code>)`` internally.
    The three budget-exhaustion pre-flight checks (combined time/tokens,
    TOCTOU re-check, timeout <= 0) are kept as three distinct
    ``trace.error('finalization_fallback', message='budget_exhausted')`` calls,
    matching the pre-refactor ``finalize_node`` so that
    ``test_finalize_skips_llm_when_time_exhausted`` and
    ``test_finalize_skips_llm_when_tokens_below_min`` (both assert exactly one
    ``finalization_fallback`` record with ``message='budget_exhausted'``) stay green.
    """

    def __init__(self, *, llm: LLMModel, trace: Trace, task: Task) -> None:
        self._llm = llm
        self._trace = trace
        self._task = task

    def _fallback(
            self,
            *,
            budget: Budget,
            history: list[dict[str, Any]],
            last_tool_ok: str | None,
            message: str,
            latency_ms: int = 0,
    ) -> tuple[str | None, Finalization]:
        """
        Record a fallback trace error and build the deterministic fallback result.

        Every fallback branch in ``run`` goes through this helper, so the
        ``trace.error`` shape (kind ``'finalization_fallback'``, machine-readable
        message) and the ``Finalization`` value are guaranteed identical across
        all paths.

        Args:
            budget (Budget): Budget for the run (by reference; only read here).
            history (list[dict[str, Any]]): Conversation history for the deterministic fallback.
            last_tool_ok (str | None): Content of the most recent successful tool result, or None.
            message (str): Machine-readable code for trace.error ``message``.
            latency_ms (int): Latency for trace.error (0 for pre-flight branches).

        Returns:
            tuple[str | None, Finalization]: (answer, finalization).
        """
        self._trace.error(
            step=budget.steps,
            kind='finalization_fallback',
            message=message,
            latency_ms=latency_ms,
            budget=budget,
        )

        answer = _deterministic_fallback(history, last_tool_ok)

        return answer, (Finalization.FALLBACK if answer is not None else Finalization.NONE)

    async def run(
            self,
            *,
            budget: Budget,
            history: list[dict[str, Any]],
            last_tool_ok: str | None,
    ) -> tuple[str | None, Finalization]:
        """
        Perform the single bounded finalization LLM call.

        All hard-budget pre-flight and fallback paths write
        ``trace.error(kind='finalization_fallback', message=<code>)`` internally.

        Args:
            budget (Budget): Budget for the run (mutated by reference).
            history (list[dict[str, Any]]): Conversation history to synthesize from.
            last_tool_ok (str | None): Content of the most recent successful tool result, or None.

        Returns:
            tuple[str | None, Finalization]: (answer, finalization).
        """
        # Hard budget pre-flight: the finalization call must not breach the remaining wall-clock or token budgets.
        if budget.remaining_seconds() <= 0 or budget.remaining_tokens() < settings.MIN_EXPECTED_TOKENS:
            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='budget_exhausted',
            )

        system_prompt = self._task.system_prompt + '\n\n' + FINALIZE_ANCHOR

        # Re-check remaining time to defend against a TOCTOU race between the budget check above and the actual call.
        remaining = budget.remaining_seconds()

        if remaining <= 0:
            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='budget_exhausted',
            )

        timeout = min(remaining, settings.LLM_DEFAULT_TIMEOUT_SECONDS)

        if timeout <= 0:
            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='budget_exhausted',
            )

        t0 = time.monotonic()

        try:
            chat = getattr(self._llm, 'chat', None)

            if chat is None:
                raise ValueError(f"{self._llm.__class__.__name__} does not support agent chat()")

            reply = await chat(history, [], timeout, system_prompt)
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.monotonic() - t0) * 1000)

            # Same helper used by LLMStepService so exceptions are not silently swallowed by the fallback path.
            # step=budget.steps keeps the association with the last executed step.
            _log_internal_exception('Finalization LLM call failed.', exc, step=budget.steps)

            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='llm_error', latency_ms=latency_ms,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Account for the call regardless of outcome: tokens and cost were consumed.
        if reply is not None:
            budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost_usd)

        # reply.tool_calls is ignored on finalization: the model was given an empty tool list, but the Responses API
        # does not enforce it. We do not propagate them into pending_tool_calls, to keep the pairing invariant intact.
        if reply is None or not reply.text.strip():
            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='empty_reply', latency_ms=latency_ms,
            )

        if reply.tool_calls:
            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='tool_calls_in_reply', latency_ms=latency_ms,
            )

        # A successful finalization may itself push us over a hard budget. Record this for observability without
        # changing the termination reason: BUDGET_STEPS remains the primary reason, and the budget snapshot
        # in trace.finish carries the actual usage.
        overrun = budget.exceeded_without_steps()

        if overrun is not None:
            self._trace.error(
                step=budget.steps,
                kind='finalization_overrun',
                message=overrun.value,
                latency_ms=latency_ms,
                budget=budget,
            )

        return reply.text, Finalization.LLM
