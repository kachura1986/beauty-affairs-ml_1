import logging

import httpx

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


def _log_internal_exception(message: str, exc: Exception, step: int | None = None) -> None:
    extra = {'step': step}

    logger.error(
        '%s | step=%s',
        message,
        step,
        exc_info=(type(exc), exc, exc.__traceback__),
        extra=extra,
    )
