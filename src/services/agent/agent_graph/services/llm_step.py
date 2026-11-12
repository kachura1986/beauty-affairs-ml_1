import asyncio
import logging
import time

from core.config import settings
from services.llm import LLMModel, Reply

from ...context import AgentContextWindow
from ...domain import Budget, TerminationReason
from ...tools import ToolRegistry
from ...trace import Trace
from .._common import (
    _classify_llm_exception,
    _log_internal_exception,
    _retry_backoff,
    _trace_llm_error,
)

logger = logging.getLogger(__name__)


class LLMStepService:
    """One bounded LLM decision step: budget pre-flight, error classification, token/cost accounting, trace."""

    def __init__(self, *, llm: LLMModel, tools: ToolRegistry, trace: Trace) -> None:
        self._llm = llm
        self._tools = tools
        self._trace = trace

    async def run(
            self, *, budget: Budget, context_window: AgentContextWindow,
    ) -> tuple[Reply | None, TerminationReason | None]:
        """Execute one LLM decision step, with at most one retry on a transient error.

        CONTRACT: ``budget.can_start_step()`` must be True — guaranteed by the only caller,
        :func:`decide_node`, via :func:`_check_budget`; the step budget is intentionally not re-checked
        here to keep a single enforcement point. Mutates ``budget`` by reference; callers must pass
        ``budget=state['budget']``, never a copy. When ``reason`` is not ``None``, ``reply`` may still be
        valid: a completed LLM call that itself pushed the run over a hard budget is preserved so the
        caller can decide whether it qualifies as a final answer. ``budget.steps`` is incremented at the
        top of every attempt — a retry is a physical LLM invocation and consumes a step.

        Args:
            budget (Budget): Budget for the request (mutated by reference).
            context_window (AgentContextWindow): Context for the request.

        Returns:
            tuple[Reply | None, TerminationReason | None]: Tuple of (reply, termination reason).
        """
        # Explicit raise, not assert: under `python -O` an assert is stripped and the violation
        # would pass silently, breaking the steps <= max_steps invariant.
        if not budget.can_start_step():
            raise RuntimeError('precondition violated: LLMStepService.run called with exhausted step budget')

        # TOCTOU guard: build_context() runs between _check_budget() and this call, so the wall-clock
        # budget may have expired in the meantime.
        remaining = budget.remaining_seconds()

        if remaining <= 0:
            return None, TerminationReason.BUDGET_TIME

        # Strict '<' on purpose: a caller that lands exactly on MIN_EXPECTED_TOKENS is still allowed to try.
        if budget.remaining_tokens() < settings.MIN_EXPECTED_TOKENS:
            logger.info(
                'Preflight token check failed | step=%s | remaining=%s | min=%s',
                budget.steps,
                budget.remaining_tokens(),
                settings.MIN_EXPECTED_TOKENS,
                extra={'step': budget.steps, 'error_code': 'preflight_tokens'},
            )
            self._trace.error(
                step=budget.steps,
                kind='preflight_tokens',
                message='budget_tokens',
                latency_ms=0,
                budget=budget,
            )

            return None, TerminationReason.BUDGET_TOKENS

        for attempt in range(2):
            budget.steps += 1

            t0 = time.monotonic()

            try:
                return await self._attempt(budget, context_window, t0)
            except Exception as exc:  # noqa: BLE001
                latency_ms = int((time.monotonic() - t0) * 1000)

                _log_internal_exception(
                    "LLM request failed." if attempt == 0 else "LLM retry failed.",
                    exc,
                    step=budget.steps,
                )

                # `llm_timeout` only when the budget is exhausted *after* the exception — not for any TimeoutError.
                if budget.remaining_seconds() <= 0:
                    _trace_llm_error(
                        trace=self._trace, step=budget.steps, kind='llm_timeout',
                        exc=exc, latency_ms=latency_ms, budget=budget,
                    )
                    return None, TerminationReason.BUDGET_TIME

                # llm_transient is emitted for every transient failure, regardless of whether a retry follows.
                kind = _classify_llm_exception(exc, llm=self._llm)
                _trace_llm_error(
                    trace=self._trace, step=budget.steps, kind=kind,
                    exc=exc, latency_ms=latency_ms, budget=budget,
                )

                if attempt == 1 or kind != 'llm_transient' or not self._can_retry(budget):
                    return None, TerminationReason.LLM_ERROR

                # Retry bookkeeping: `llm_retry` is traced at the retry attempt's step number
                # (`budget.steps + 1`), preserving the pre-refactor order — increment, then trace.
                backoff = _retry_backoff(budget.remaining_seconds())
                _trace_llm_error(
                    trace=self._trace, step=budget.steps + 1, kind='llm_retry',
                    exc=exc, latency_ms=latency_ms, budget=budget,
                )

                if backoff > 0:
                    await asyncio.sleep(backoff)

        raise RuntimeError("unreachable")  # pragma: no cover

    async def _attempt(
            self, budget: Budget, context_window: AgentContextWindow, t0: float,
    ) -> tuple[Reply | None, TerminationReason | None]:
        """One bounded LLM call attempt; ``chat()`` exceptions propagate to :meth:`run` for classification.

        CONTRACT: the TOCTOU guard (``remaining <= 0``) returns a sentinel tuple instead of raising.

        ACCOUNTING FAILURE (``budget.record_llm``): hard-fail with ``LLM_ERROR``, no retry — a retry
        would invoke ``chat()`` a second time while the first call was already billed, doubling
        token/cost accounting. The failure is traced once as ``llm_error`` with
        ``message=type(exc).__name__``.

        TRADE-OFF: on accounting failure the already-received ``reply`` is intentionally dropped
        (``return None, LLM_ERROR``). Returning it as ``answer`` would yield an inconsistent trace
        (``answer_chars > 0`` but ``tokens = 0``). A future design that returns the reply must add an
        ``accounting_failed=true`` marker.

        TRACE FAILURE (``trace.step``): best-effort — logged, then swallowed. The trace is
        diagnostic; a failed trace write must not affect the run, let alone trigger a retry.

        Args:
            budget (Budget): Budget for the request (mutated by reference on success).
            context_window (AgentContextWindow): Context for the request.
            t0 (float): Monotonic timestamp captured by the caller before invoking this helper;
                used to compute ``latency_ms`` for the trace record.

        Returns:
            tuple[Reply | None, TerminationReason | None]:
                - ``(reply, None)``: successful call within budget.
                - ``(reply, reason)``: successful call that pushed the run over a hard budget; the
                  caller decides if the reply qualifies as a final answer.
                - ``(None, BUDGET_TIME)``: TOCTOU guard — ``chat`` was not invoked.
                - ``(None, LLM_ERROR)``: accounting failure — ``chat`` succeeded, usage could not
                  be recorded, retry intentionally suppressed.

        Raises:
            Exception: Any exception raised by ``chat`` is propagated unchanged.
        """
        remaining = budget.remaining_seconds()

        if remaining <= 0:
            return None, TerminationReason.BUDGET_TIME

        timeout = min(remaining, settings.LLM_DEFAULT_TIMEOUT_SECONDS)

        chat = getattr(self._llm, 'chat', None)

        if chat is None:
            raise ValueError(f"{self._llm.__class__.__name__} does not support agent chat()")

        # (A) chat()
        reply = await chat(context_window.messages, self._tools.schemas(), timeout, context_window.system)

        latency_ms = int((time.monotonic() - t0) * 1000)

        # (B) record_llm()
        try:
            budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost)
        except Exception as acc_exc:  # noqa: BLE001
            _log_internal_exception("Failed to record LLM usage.", acc_exc, step=budget.steps)
            _trace_llm_error(
                trace=self._trace, step=budget.steps, kind='llm_error',
                exc=acc_exc, latency_ms=latency_ms, budget=budget,
            )
            return None, TerminationReason.LLM_ERROR

        # (C) trace.step()
        try:
            self._trace.step(
                budget.steps,
                context_window.messages,
                reply,
                budget,
                latency_ms,
                system_prompt=context_window.system,
            )
        except Exception as trace_exc:  # noqa: BLE001
            _log_internal_exception("Failed to write step trace.", trace_exc, step=budget.steps)

        # A completed call may push us over budget. Keep the reply — dropping it would turn
        # a finished step into an empty AgentResult. decide_node decides if it's a final answer.
        reason = budget.exceeded_without_steps()

        return reply, reason

    def _can_retry(self, budget: Budget) -> bool:
        """Return True if a retry is allowed by every hard limit.

        CONTRACT: the ``remaining_tokens`` check is defensive — :meth:`run` already verified it in
        preflight, and a failed attempt does not consume tokens (``record_llm`` runs only on success).
        It guards against future changes where ``record_llm`` might run before a failure is raised.

        Args:
            budget (Budget): Budget for the request (read-only).

        Returns:
            bool: True if a retry is allowed by every hard limit.
        """
        if not budget.can_start_step():
            return False

        remaining = budget.remaining_seconds()
        backoff = _retry_backoff(remaining)

        # The retry needs time to back off AND time to actually run the second chat call;
        # LLM_RETRY_MIN_TIMEOUT_SECONDS reserves the latter.
        if remaining <= backoff + settings.LLM_RETRY_MIN_TIMEOUT_SECONDS:
            return False

        return budget.remaining_tokens() >= settings.MIN_EXPECTED_TOKENS
