import logging
import time

from core.config import settings
from services.llm import LLMModel, Reply

from ...context import AgentContextWindow
from ...domain import Budget, TerminationReason
from ...tools import ToolRegistry
from ...trace import Trace
from .._common import _is_transient_llm_error, _log_internal_exception

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
        """Execute exactly one LLM decision step.

        CONTRACT: ``budget.can_start_step()`` must be True — guaranteed by the only caller,
        :func:`decide_node`, via :func:`_check_budget`; the step budget is intentionally not re-checked
        here to keep a single enforcement point. Mutates ``budget`` by reference; callers must pass
        ``budget=state['budget']``, never a copy. When ``reason`` is not ``None``, ``reply`` may still be valid:
        a completed LLM call that itself pushed the run over a hard budget is preserved so the caller can decide
        whether it qualifies as a final answer.

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

        budget.steps += 1

        t0 = time.monotonic()

        try:
            chat = getattr(self._llm, 'chat', None)

            if chat is None:
                raise ValueError(f"{self._llm.__class__.__name__} does not support agent chat()")

            timeout = min(remaining, settings.LLM_DEFAULT_TIMEOUT_SECONDS)
            reply = await chat(context_window.messages, self._tools.schemas(), timeout, context_window.system)

        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.monotonic() - t0) * 1000)

            _log_internal_exception("LLM request failed.", exc, step=budget.steps)

            # `llm_timeout` only when the budget is exhausted *after* the exception — not for any TimeoutError.
            if budget.remaining_seconds() <= 0:
                self._trace.error(
                    step=budget.steps,
                    kind='llm_timeout',
                    message=type(exc).__name__,
                    latency_ms=latency_ms,
                    budget=budget,
                )

                return None, TerminationReason.BUDGET_TIME

            # `kind` distinguishes transient vs permanent for post-mortem; TerminationReason is unchanged.
            kind = 'llm_transient' if _is_transient_llm_error(exc) else 'llm_error'
            self._trace.error(
                step=budget.steps,
                kind=kind,
                message=type(exc).__name__,
                latency_ms=latency_ms,
                budget=budget,
            )

            return None, TerminationReason.LLM_ERROR

        latency_ms = int((time.monotonic() - t0) * 1000)

        budget.record_llm(reply.in_tokens, reply.out_tokens, reply.cost_usd)

        self._trace.step(
            budget.steps,
            context_window.messages,
            reply,
            budget,
            latency_ms,
            system_prompt=context_window.system,
        )

        # A completed call may push us over budget. Keep the reply — dropping it would turn
        # a finished step into an empty AgentResult. decide_node decides if it's a final answer.
        reason = budget.exceeded_without_steps()

        if reason:
            return reply, reason

        return reply, None
