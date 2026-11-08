import asyncio
import time
from typing import Any

from core.config import settings
from services.llm import LLMModel, Reply

from ...domain import Budget, Finalization, Task
from ...trace import Trace
from .._common import (
    FINALIZE_ANCHOR,
    _classify_llm_exception,
    _log_internal_exception,
    _retry_backoff,
    _trace_llm_error,
)
from .._fallback import _deterministic_fallback


class FinalizeService:
    """
    Single bounded finalization LLM call after max_steps is exhausted.

    CONTRACT (enforced by ``finalize_node``): reason is ``BUDGET_STEPS``, ``needs_finalization`` True,
    ``answer`` None, policy enabled; returns ``(answer, finalization)`` without touching
    ``termination_reason``/``needs_finalization``. Every fallback writes
    ``trace.error('finalization_fallback', message=<code>)``; success writes ``t='finalize'``.
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
        Sole fallback exit: emits a uniform ``trace.error`` and builds the deterministic result.

        Args:
            budget (Budget): Budget for the run (read-only).
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
        Perform the bounded finalization call with at most one transient retry; on failure return the fallback.

        CONTRACT: cost exhaustion is NOT a pre-flight — a free model can still answer at zero cost
        (overrun then recorded as ``trace.error('finalization_overrun')``). ``budget.steps`` is NOT
        incremented on retry (finalize runs out-of-steps by contract). ``finalization_fallback`` is
        emitted only on the terminal path; the retry path emits only ``finalization_retry``, so a
        successful retry leaves zero ``finalization_fallback`` events (P2 #5).

        Args:
            budget (Budget): Budget for the run (mutated by reference).
            history (list[dict[str, Any]]): Conversation history to synthesize from.
            last_tool_ok (str | None): Content of the most recent successful tool result, or None.

        Returns:
            tuple[str | None, Finalization]: (answer, finalization).
        """
        # Hard budget pre-flight: the finalization call must not breach remaining wall-clock or token budgets.
        if budget.remaining_seconds() <= 0 or budget.remaining_tokens() < settings.MIN_EXPECTED_TOKENS:
            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message='budget_exhausted',
            )

        system_prompt = self._task.system_prompt + '\n\n' + FINALIZE_ANCHOR

        for attempt in range(2):
            t0 = time.monotonic()

            try:
                reply = await self._call(budget, history, system_prompt)
                latency_ms = int((time.monotonic() - t0) * 1000)
                break
            except Exception as exc:  # noqa: BLE001
                latency_ms = int((time.monotonic() - t0) * 1000)

                # Same helper as LLMStepService so exceptions are not silently swallowed by the fallback path.
                _log_internal_exception(
                    'Finalization LLM call failed.' if attempt == 0 else 'Finalization retry failed.',
                    exc,
                    step=budget.steps,
                )

                # Mirror LLMStepService.run: `llm_timeout` only when the budget is exhausted *after*
                # the exception. Otherwise classify transient vs permanent for post-mortem.
                if budget.remaining_seconds() <= 0:
                    return self._fallback(
                        budget=budget, history=history, last_tool_ok=last_tool_ok,
                        message='llm_timeout', latency_ms=latency_ms,
                    )

                classification = _classify_llm_exception(exc)

                if attempt == 1 or classification != 'llm_transient' or not self._can_retry(budget):
                    return self._fallback(
                        budget=budget, history=history, last_tool_ok=last_tool_ok,
                        message=classification, latency_ms=latency_ms,
                    )

                # P2 #5: `finalization_fallback` is reserved for the terminal path
                # (via `_fallback`). On the retry path we emit only
                # `finalization_retry`, so a successful retry leaves zero
                # `finalization_fallback` events — the alert "one fallback per
                # failure" holds. Finalize runs out-of-steps by contract, so
                # budget.steps is intentionally NOT incremented here — the retry
                # is bounded by wall-clock and tokens only (see _can_retry).
                backoff = _retry_backoff(budget.remaining_seconds())
                _trace_llm_error(
                    trace=self._trace, step=budget.steps, kind='finalization_retry',
                    exc=exc, latency_ms=latency_ms, budget=budget,
                )

                if backoff > 0:
                    await asyncio.sleep(backoff)
        else:
            raise RuntimeError("unreachable")  # pragma: no cover

        # Account for the call regardless of outcome: tokens and cost were consumed.
        if reply is not None:
            budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost_usd)

        # tool_calls are ignored on finalization: the model was given an empty tool list, but the
        # Responses API does not enforce it. Propagating them into pending_tool_calls would break
        # the execute_node pairing invariant.
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

        self._trace.finalize(
            budget.steps,
            history,
            reply,
            budget,
            latency_ms,
            system_prompt=system_prompt,
        )

        # A successful finalization may itself push us over a hard budget. Record it for observability
        # without changing the termination reason: BUDGET_STEPS remains the primary reason.
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

    async def _call(
            self, budget: Budget, history: list[dict[str, Any]], system_prompt: str,
    ) -> Reply:
        """
        One bounded finalization LLM call; exceptions propagate so :meth:`run` can classify and retry.

        CONTRACT: caller guarantees ``remaining_seconds() > 0`` (hard pre-flight), so no TOCTOU
        guard is needed.

        Args:
            budget (Budget): Budget for the run (read-only).
            history (list[dict[str, Any]]): Conversation history.
            system_prompt (str): System prompt with the finalization anchor.

        Returns:
            Reply: The model reply.

        Raises:
            Exception: Any exception raised by ``chat``.
        """
        timeout = min(budget.remaining_seconds(), settings.LLM_DEFAULT_TIMEOUT_SECONDS)

        chat = getattr(self._llm, 'chat', None)

        if chat is None:
            raise ValueError(f"{self._llm.__class__.__name__} does not support agent chat()")

        return await chat(history, [], timeout, system_prompt)

    def _can_retry(self, budget: Budget) -> bool:
        """
        Return True if a retry is allowed by every applicable hard limit.

        CONTRACT: finalize runs out-of-steps by design (``finalize_on_max_steps`` carve-out), so
        ``can_start_step`` is intentionally NOT checked; ``budget.steps`` is not incremented on
        retry either — bounded by wall-clock and tokens.

        Args:
            budget (Budget): Budget for the run (read-only).

        Returns:
            bool: True if a retry is allowed by every applicable hard limit.
        """
        remaining = budget.remaining_seconds()
        backoff = _retry_backoff(remaining)

        # The retry needs time to back off AND time to actually run the second chat call;
        # LLM_RETRY_MIN_TIMEOUT_SECONDS reserves the latter.
        if remaining <= backoff + settings.LLM_RETRY_MIN_TIMEOUT_SECONDS:
            return False

        # Defensive: run() already verified remaining_tokens >= MIN_EXPECTED_TOKENS in pre-flight,
        # and a failed attempt does not consume tokens (record_llm runs only on success). This
        # guards against future changes where record_llm might run before a failure is raised.
        return budget.remaining_tokens() >= settings.MIN_EXPECTED_TOKENS
