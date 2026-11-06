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
    Single bounded finalization LLM call after max_steps is exhausted.

    Preconditions (enforced by ``finalize_node``): ``BUDGET_STEPS`` reason,
    ``needs_finalization`` True, ``answer`` None, policy enabled.

    Returns ``(answer, finalization)``. Does not touch ``termination_reason``
    (overrun → ``trace.error('finalization_overrun')``, reason stays
    ``BUDGET_STEPS``) or ``needs_finalization`` — that is the node's contract.

    Fallbacks write ``trace.error('finalization_fallback', message=<code>)``. The pre-flight check covers
    wall-clock and token budgets and emits at most one ``message='budget_exhausted'`` record; an LLM
    exception is classified as ``'llm_timeout'`` when the wall-clock budget is exhausted after the failure,
    otherwise ``'llm_error'`` (mirrors ``LLMStepService.run``).

    A successful call writes ``trace.finalize(...)`` (``t='finalize'``), so post-mortem can
    separate the finalization cost/latency from decision steps (``t='step'``); a hard non-step
    budget overrun is additionally logged as ``trace.error('finalization_overrun', ...)``.
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

        Every fallback branch in ``run`` goes through this helper, so the ``trace.error`` shape (kind
        ``'finalization_fallback'``, machine-readable message) and the ``Finalization`` value are
        guaranteed identical across all paths.

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

        All hard-budget pre-flight and fallback paths write ``trace.error(kind='finalization_fallback', message=<code>)``
        internally. The wall-clock and token budgets are checked once before the call; no second re-check is
        needed because no I/O occurs between the check and the call. A successful call writes a ``t='finalize'``
        record with model, token counts, cost, latency, and ``history_msgs``, so post-mortem can attribute
        the finalization call separately from decision steps.

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

        # Cap the call by the remaining wall-clock budget. No second `<= 0` check: the pre-flight above
        # already guarantees remaining > 0, and no I/O or await happens between that check and this line.
        timeout = min(budget.remaining_seconds(), settings.LLM_DEFAULT_TIMEOUT_SECONDS)

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

            # Distinguish a wall-clock timeout from a real LLM error, mirroring LLMStepService.run: `llm_timeout`
            # only when the budget is exhausted *after* the exception, not for any TimeoutError.
            message = 'llm_timeout' if budget.remaining_seconds() <= 0 else 'llm_error'

            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message=message, latency_ms=latency_ms,
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

        # Successful finalization: record the call so post-mortem can separate its cost and latency
        # from decision steps (`t='step'`). Mirrors `Trace.step`, tagged `t='finalize'`.
        self._trace.finalize(
            budget.steps,
            history,
            reply,
            budget,
            latency_ms,
            system_prompt=system_prompt,
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
