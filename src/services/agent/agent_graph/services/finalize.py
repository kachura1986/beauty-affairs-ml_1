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

    CONTRACT (enforced by ``finalize_node``): reason is ``BUDGET_STEPS``, ``needs_finalization`` True,
    ``answer`` None, policy enabled. Returns ``(answer, finalization)`` and does not touch
    ``termination_reason`` or ``needs_finalization`` — that is the node's contract.
    All fallbacks write ``trace.error('finalization_fallback', message=<code>)``; a successful call
    writes ``t='finalize'`` so post-mortem can separate its cost/latency from decision steps.
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

        Every fallback branch goes through here, so the ``trace.error`` shape and the
        ``Finalization`` value are identical across all paths.

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

        CONTRACT: cost exhaustion is NOT a pre-flight — ``remaining_cost_usd() == 0`` only means
        accumulated spend hit the limit; a free model can still answer at zero cost, so the call
        proceeds (an overrun is then recorded as ``trace.error('finalization_overrun')``).

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

        # No second `<= 0` check: the pre-flight above guarantees remaining > 0, and no I/O or await
        # happens between that check and this line.
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
            _log_internal_exception('Finalization LLM call failed.', exc, step=budget.steps)

            # Mirror LLMStepService.run: `llm_timeout` only when the budget is exhausted *after*
            # the exception, not for any TimeoutError.
            message = 'llm_timeout' if budget.remaining_seconds() <= 0 else 'llm_error'

            return self._fallback(
                budget=budget, history=history, last_tool_ok=last_tool_ok,
                message=message, latency_ms=latency_ms,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Account for the call regardless of outcome: tokens and cost were consumed.
        if reply is not None:
            budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost_usd)

        # tool_calls are ignored on finalization: the model was given an empty tool list, but the
        # Responses API does not enforce it. Do not propagate them into pending_tool_calls — doing
        # so would break the execute_node pairing invariant.
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
