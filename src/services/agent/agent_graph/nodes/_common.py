import logging

logger = logging.getLogger(__name__)

# Appended to the task system prompt for the single finalization LLM call that
# runs after max_steps has been exhausted. Tools are disabled for this call.
FINALIZE_ANCHOR = (
    'You have reached the final step. Provide the best final answer now using '
    'the information already collected. Do not call any tools.'
)


def _log_internal_exception(message: str, exc: Exception, step: int | None = None) -> None:
    extra = {'step': step}

    logger.error(
        '%s | step=%s',
        message,
        step,
        exc_info=(type(exc), exc, exc.__traceback__),
        extra=extra,
    )
