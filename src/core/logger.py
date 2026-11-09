import logging
import os
import sys
from functools import lru_cache
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from core.config import settings

# Base JSON event fields (never dropped)
_BASE_KEYS: frozenset[str] = frozenset({
    'timestamp',
    'level',
    'logger',
    'event',
    'exception',
    'stack',
})

# Fields the shipped ErrorHandler itself relies on — not user configurable
_INTERNAL_KEYS: tuple[str, ...] = ('error_code',)

# Fields allowed to pass through extra=... into JSON
_EXTRA_WHITELIST: tuple[str, ...] = (
    'run_id',
    'step',
    'tool',
    'latency_ms',
    'log_file',
)

_ALLOWED_KEYS = _BASE_KEYS | frozenset(_INTERNAL_KEYS) | frozenset(_EXTRA_WHITELIST)


def _add_whitelisted_extras(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """
    Copies whitelisted fields from LogRecord.__dict__ into event_dict.

    Needed for stdlib logs issued with extra=... such as: logging.getLogger(...).info('...', extra={'run_id': ...})

    For native structlog calls there is no _record, so this is a no-op;
    the final _filter_event_dict still enforces the whitelist.

    Projected keys are _INTERNAL_KEYS plus the user-editable _EXTRA_WHITELIST. The internal keys are copied
    unconditionally: the shipped ErrorHandler relies on error_code, and that contract must not depend on whether the
    user whitelist is empty.

    Args:
        _logger (Any): structlog logger (unused).
        _method_name (str): Logging method name (unused).
        event_dict (EventDict): Current structlog event dictionary.

    Returns:
        EventDict: The same event dictionary, enriched with whitelisted extras.
    """
    record = event_dict.get('_record')
    if record is None:
        return event_dict

    for key in (*_INTERNAL_KEYS, *_EXTRA_WHITELIST):
        value = record.__dict__.get(key)
        if value is not None:
            event_dict[key] = value

    return event_dict


def _filter_event_dict(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """
    Returns a new event_dict containing only allowed keys.

    Pure function: does not mutate the input, which is safer if context_class
    is ever changed from dict to an ordered/custom mapping.

    Drops:
        * internal LogRecord attributes (args, exc_info, msg, ...);
        * structlog meta keys (_record, _from_structlog);
        * arbitrary extra fields (e.g. SecretStr, large payloads);
        * projected fields (internal keys + user whitelist) whose value is None.

    Args:
        _logger: structlog logger (unused).
        _method_name: Logging method name (unused).
        event_dict: Current structlog event dictionary.

    Returns:
        EventDict: A new dict with only allowed keys and non-None values.
    """
    return {
        key: value
        for key, value in event_dict.items()
        if key in _ALLOWED_KEYS
           and not (key in (*_INTERNAL_KEYS, *_EXTRA_WHITELIST) and value is None)
    }


def _build_shared_processors() -> list[Processor]:
    """
    Builds the processor chain shared by foreign stdlib logs.

    The final JSONRenderer is intentionally excluded here: rendering is done
    exactly once, inside ProcessorFormatter.processors.

    Returns:
        list[Processor]: Ordered list of pre-render processors.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt='iso', utc=True, key='timestamp'),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_whitelisted_extras,
    ]


def _is_ours(handler: logging.Handler) -> bool:
    """
    Return True if the handler was installed by setup_logging.

    Foreign handlers (pytest plugins, host applications, etc.) must not
    prevent our handlers from being installed, and must not be treated
    as evidence that setup already ran.

    Args:
        handler (logging.Handler): Any logging handler attached to root.

    Returns:
        bool: True if the handler carries the _structlog_handler marker.
    """
    return getattr(handler, '_structlog_handler', False)


def setup_logging(logs_dir: Path, log_filename: str = 'app.log') -> logging.Logger:
    """
    Configures structured JSON logging for the application.

    Logs are written to both a file and the console, with log rotation every midnight.
    The "logs" directory is automatically created if it doesn't exist. Older logs are kept for 7 days.

    Fields passed via extra=... are projected onto the JSON event when their
    names are in the whitelist: run_id, step, tool, error_code, latency_ms, log_file.

    Foreign handlers already attached to the root logger (e.g. pytest plugins,
    embedding applications) are left untouched: setup_logging only appends its
    own file and console handlers. Idempotency is defined as "our handlers are
    present", not "root has no handlers".

    logger.setLevel(INFO) and the noisy-logger quieting are applied when our
    handlers are installed. A second call early-returns, so they are not
    re-applied — that is deliberate: our handlers already exist, and the
    loggers are already quieted.

    Args:
        logs_dir (Path): Directory where logs will be stored.
        log_filename (str, optional): Log file name (default: "app.log").

    Returns:
        logging.Logger: Configured root logger with JSON handlers.
    """
    logger = logging.getLogger()

    # Prevent duplicate installation of *our* handlers. Foreign handlers (plugins, host app) must be ignored,
    # not treated as evidence that setup already ran.
    if any(_is_ours(h) for h in logger.handlers):
        return logger

    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, log_filename)

    # Configure structlog to route through stdlib logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt='iso', utc=True, key='timestamp'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _add_whitelisted_extras,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Single JSON formatter; JSONRenderer lives only here (no double encoding)
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_build_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _filter_event_dict,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )

    # Create rotating file handler
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8',
    )
    file_handler._structlog_handler = True
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler._structlog_handler = True
    console_handler.setFormatter(formatter)

    # Configure logging
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Suppress verbose HTTP client logs (Azure SDK, httpx, httpcore)
    for noisy_logger in (
            "azure.core.pipeline.policies.http_logging_policy",
            "httpx",
            "httpcore",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger.info(
        f"Logging setup complete. Logs are stored in {log_file}",
        extra={'log_file': log_file},
    )

    return logger


@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    return setup_logging(settings.LOGS_DIR, 'app.log')
