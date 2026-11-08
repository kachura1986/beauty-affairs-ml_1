import logging
from typing import Literal

from core.config import settings
from services.llm import LLMModel

from ..domain import Budget
from ..trace import Trace

logger = logging.getLogger(__name__)

FINALIZE_ANCHOR = (
    'You have reached the final step. Provide the best final answer now using '
    'the information already collected. Do not call any tools.'
)

TransientKind = Literal["llm_transient", "llm_error"]


def _classify_llm_exception(exc: Exception, *, llm: LLMModel) -> TransientKind:
    """Return the trace-friendly kind for an LLM failure.

    CONTRACT: classification is delegated to ``LLMModel.is_transient_error`` —
    the agent layer must not depend on provider internals (OpenAI SDK, httpx,
    status codes). Both ``LLMStepService`` and ``FinalizeService`` go through
    this function, so a policy change lives in the LLM implementation, not in
    four call sites.

    Args:
        exc (Exception): Exception raised by an LLM call.
        llm (LLMModel): The model that raised ``exc`` — owns the classification.

    Returns:
        TransientKind: ``'llm_transient'`` or ``'llm_error'``.
    """
    return "llm_transient" if llm.is_transient_error(exc) else "llm_error"


def _trace_llm_error(
        *,
        trace: Trace,
        step: int,
        kind: str,
        exc: Exception,
        latency_ms: int,
        budget: Budget,
) -> None:
    """Emit a trace.error event for an LLM failure with ``message=type(exc).__name__``.

    CONTRACT: ``message`` is the exception type name, never ``str(exc)`` — see
    :meth:`Trace.error`. Risk reduction only, not enforcement: call sites may bypass
    the helper.

    CARVE-OUT: ``finalization_fallback`` is NOT covered — its ``message`` is a
    machine-readable code, not ``type(exc).__name__``; it is emitted by
    ``FinalizeService._fallback`` and by the retry branch in ``FinalizeService.run``.

    Args:
        trace (Trace): Trace sink.
        step (int): Step number.
        kind (str): Trace kind (``llm_transient``, ``llm_error``, ``llm_timeout``,
            ``llm_retry``, ``finalization_retry``).
        exc (Exception): Exception to derive ``message`` from.
        latency_ms (int): Call latency in milliseconds.
        budget (Budget): Current budget.
    """
    trace.error(
        step=step,
        kind=kind,
        message=type(exc).__name__,
        latency_ms=latency_ms,
        budget=budget,
    )


def _retry_backoff(remaining_seconds: float) -> float:
    """Return the retry backoff derived from the remaining wall-clock time.

    CONTRACT: capped at ``remaining_seconds / 4``, so a small remaining budget is not
    entirely consumed by the backoff.

    Args:
        remaining_seconds (float): Remaining wall-clock budget, in seconds.

    Returns:
        float: Backoff duration in seconds, never exceeding ``remaining_seconds / 4``.
    """
    return min(settings.LLM_RETRY_BACKOFF_SECONDS, remaining_seconds / 4)


def _log_internal_exception(message: str, exc: Exception, step: int | None = None) -> None:
    extra = {'step': step}

    logger.error(
        '%s | step=%s',
        message,
        step,
        exc_info=(type(exc), exc, exc.__traceback__),
        extra=extra,
    )
