import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dependencies import get_logger

P = ParamSpec('P')
R = TypeVar('R')


def retry_strategy(
        attempts: int = 3,
        min_wait: int = 2,
        max_wait: int = 5,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Wrapper for retry strategy.
    Supports both synchronous and asynchronous functions natively.

    Args:
        attempts (int, optional): Maximum number of retry attempts. Defaults to 3.
        min_wait (int, optional): Minimum wait time in seconds. Defaults to 2.
        max_wait (int, optional): Maximum wait time in seconds. Defaults to 5.
        exceptions (tuple, optional): Exceptions to catch. Defaults to (Exception,).

    Returns:
        Callable: Decorated function.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        logger = get_logger()

        return retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True,
        )(func)

    return decorator
