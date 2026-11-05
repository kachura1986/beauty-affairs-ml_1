import logging
import time

from core.config import settings
from services.llm import LLMModel, Reply

from ...context import AgentContextWindow
from ...domain import Budget, TerminationReason
from ...tools import ToolRegistry
from ...trace import Trace
from .._common import _log_internal_exception

logger = logging.getLogger(__name__)


class LLMStepService:
    """
    One bounded LLM decision step with budget pre-flight, error classification,
    token/cost accounting, and trace recording.

    The service is stateless with respect to its own fields; it mutates the
    ``budget`` argument by reference (``steps``, ``tokens``, ``cost_usd``)
    exactly as the pre-refactor ``_request_llm`` did. Callers must pass
    ``budget=state['budget']``, never a copy.

    ``__init__`` accepts a ``ToolRegistry``, not a precomputed schema list, so the
    schema snapshot is taken inside ``run`` at call time (same as the
    pre-refactor ``_request_llm``; preserves
    ``test_request_llm_success``).
    """

    def __init__(self, *, llm: LLMModel, tools: ToolRegistry, trace: Trace) -> None:
        self._llm = llm
        self._tools = tools
        self._trace = trace

    async def run(
            self, *, budget: Budget, context_window: AgentContextWindow,
    ) -> tuple[Reply | None, TerminationReason | None]:
        """
        Execute exactly one LLM decision step.

        Preconditions:
            ``budget.can_start_step()`` is True. This is guaranteed by the only caller,
            :func:`decide_node`, via :func:`_check_budget`. The step budget is intentionally
            not re-checked here to keep a single enforcement point.

        Args:
            budget (Budget): Budget for the request (mutated by reference).
            context_window (AgentContextWindow): Context for the request.

        Returns:
            tuple[Reply | None, TerminationReason | None]: Tuple of (reply, termination reason).
            When ``reason`` is not ``None``, ``reply`` may still be valid: a completed LLM call that itself
            pushed the run over a hard budget is preserved so the caller can decide whether it qualifies
            as a final answer. This is not a contract violation — it is a retained reply, not a missing one.
        """
        # Fail loud if the precondition is violated. Under `python -O` this assert is
        # stripped; the docstring precondition above remains the authoritative contract.
        assert budget.can_start_step(), (
            'precondition violated: LLMStepService.run called with exhausted step budget'
        )

        # TOCTOU guard: build_context() runs between _check_budget() and this call, so the wall-clock budget may have
        # expired in the meantime. Limit the request by the remaining wall-clock budget.
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

            if budget.remaining_seconds() <= 0:
                self._trace.error(
                    step=budget.steps,
                    kind='llm_timeout',
                    message=type(exc).__name__,
                    latency_ms=latency_ms,
                    budget=budget,
                )

                return None, TerminationReason.BUDGET_TIME

            self._trace.error(
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
