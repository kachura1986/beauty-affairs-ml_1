import logging

import httpx

from core.config import settings

from ..tools.errors import is_temporary_failure

logger = logging.getLogger(__name__)

# Appended to the task system prompt for the single finalization LLM call that
# runs after max_steps has been exhausted. Tools are disabled for this call.
FINALIZE_ANCHOR = (
    'You have reached the final step. Provide the best final answer now using '
    'the information already collected. Do not call any tools.'
)


def _is_transient_llm_error(exc: Exception) -> bool:
    """Classify an LLM exception as transient (retryable) or permanent.

    ``APITimeoutError`` without ``__cause__ = httpx.TimeoutException`` falls through to permanent — conservative
    by design: a dropped cause must never trigger a retry of an already-consumed budget.

    Args:
        exc (Exception): Exception raised by an LLM call.

    Returns:
        bool: True when the error is considered retryable.
    """
    # APITimeoutError is wrapped with httpx.ReadTimeout as __cause__ (holds for SDK built on httpx >= 0.23).
    if isinstance(exc.__cause__, httpx.TimeoutException):
        return False

    if is_temporary_failure(exc):
        return True

    status = getattr(exc, 'status_code', None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True

    cause = exc.__cause__
    return cause is not None and is_temporary_failure(cause)


def _retry_backoff(remaining_seconds: float) -> float:
    """Return the retry backoff derived from the remaining wall-clock time.

    Capped at ``remaining_seconds / 4`` so the retry itself still has time to run
    after the sleep; without the cap, a small remaining budget would be entirely
    consumed by the backoff.

    Shared by :class:`LLMStepService` (decision-step retry) and
    :class:`FinalizeService` (finalization retry) so both apply the same formula.

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
