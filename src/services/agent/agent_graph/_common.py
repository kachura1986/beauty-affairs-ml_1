import logging
from typing import Literal

import httpx

from core.config import settings

from ..domain import Budget
from ..tools.errors import is_temporary_failure
from ..trace import Trace

logger = logging.getLogger(__name__)

FINALIZE_ANCHOR = (
    'You have reached the final step. Provide the best final answer now using '
    'the information already collected. Do not call any tools.'
)


def _is_transient_llm_error(exc: Exception) -> bool:
    """Classify an LLM exception as transient (retryable) or permanent.

    CONTRACT: ``APITimeoutError`` without ``__cause__ = httpx.TimeoutException`` is permanent —
    a dropped cause must never trigger a retry of an already-consumed budget.

    Args:
        exc (Exception): Exception raised by an LLM call.

    Returns:
        bool: True when the error is considered retryable.
    """
    # SDK wraps APITimeoutError with httpx.ReadTimeout as __cause__ (holds for httpx >= 0.23).
    if isinstance(exc.__cause__, httpx.TimeoutException):
        return False

    if is_temporary_failure(exc):
        return True

    status = getattr(exc, 'status_code', None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True

    cause = exc.__cause__
    return cause is not None and is_temporary_failure(cause)


TransientKind = Literal["llm_transient", "llm_error"]


def _classify_llm_exception(exc: Exception) -> TransientKind:
    """Single source of truth for the transient/permanent split in trace events.

    CONTRACT: both :class:`LLMStepService` and :class:`FinalizeService` must go through it,
    so a change to the classification policy is made here once, not in four call sites.

    Args:
        exc (Exception): Exception raised by an LLM call.

    Returns:
        TransientKind: ``'llm_transient'`` or ``'llm_error'``.
    """
    return "llm_transient" if _is_transient_llm_error(exc) else "llm_error"


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
