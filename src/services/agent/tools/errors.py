
import logging
import math
import traceback
from typing import Any

from .tool import ToolError, ToolErrorCode, ToolResult

logger = logging.getLogger(__name__)

_TEMPORARY_EXCEPTION_NAMES = frozenset({
    'TimeoutException',
    'ConnectTimeout',
    'ReadTimeout',
    'WriteTimeout',
    'PoolTimeout',
    'ConnectError',
    'NetworkError',
    'TransportError',
    'ClientConnectionError',
    'ClientConnectorError',
    'ServerDisconnectedError',
    'ServiceRequestError',
    'ServiceResponseError',
})


def is_temporary_failure(exc: Exception) -> bool:
    """
    Determine whether an exception represents a temporary failure.

    Args:
        exc (Exception): Exception raised during tool execution.

    Returns:
        bool: True if the exception is considered temporary.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    return any(cls.__name__ in _TEMPORARY_EXCEPTION_NAMES for cls in type(exc).__mro__)


def retry_after_seconds(exc: Exception) -> float | None:
    """
    Return the delay before retrying, if available.

    Args:
        exc (Exception): Exception raised during tool execution.

    Returns:
        float | None: Non-negative retry delay in seconds, or None if unavailable.
    """
    value = getattr(exc, 'retry_after', None)

    if value is None:
        response = getattr(exc, 'response', None)
        headers = getattr(response, 'headers', None)

        if headers is not None:
            value = headers.get('Retry-After')
            if value is None:
                value = headers.get('retry-after')

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(value):
        return None

    return max(0.0, value)


def log_exception(message: str, tool_name: str, exc: Exception) -> None:
    """
    Log an exception with structured tool execution context.

    Args:
        message (str): Log message describing the failure.
        tool_name (str): Name of the tool that raised the exception.
        exc (Exception): Exception raised during tool execution.
    """
    logger.error(
        message,
        extra={
            'tool_name': tool_name,
            'exception_type': type(exc).__name__,
            'stack': traceback.format_tb(exc.__traceback__),
        },
    )


def validation_hint(errors: list[dict[str, Any]]) -> str:
    """
    Convert Pydantic validation errors into a hint for the LLM.

    Args:
        errors (list[dict[str, Any]]): List of validation errors returned by Pydantic.

    Returns:
        str: Human-readable instructions for correcting the arguments.
    """
    hints = []

    for error in errors:
        location = '.'.join(str(x) for x in error.get('loc', ()))
        field = f"Field '{location}'" if location else 'Arguments'
        error_type = error.get('type', '')
        ctx = error.get('ctx') or {}

        if error_type == 'missing':
            hint = f"Provide the required field '{location}'."

        elif error_type == 'extra_forbidden':
            hint = f"Field '{location}' is not supported by the schema; remove it."

        elif error_type == 'string_too_short':
            limit = ctx.get('min_length')
            hint = f"{field}: provide a string at least {limit} characters long."

        elif error_type == 'string_too_long':
            limit = ctx.get('max_length')
            hint = f"{field}: shorten the string to at most {limit} characters."

        elif error_type == 'greater_than_equal':
            limit = ctx.get('ge')
            hint = f"{field}: the value must be greater than or equal to {limit}."

        elif error_type == 'less_than_equal':
            limit = ctx.get('le')
            hint = f"{field}: the value must be less than or equal to {limit}."

        elif error_type == 'greater_than':
            limit = ctx.get('gt')
            hint = f"{field}: the value must be greater than {limit}."

        elif error_type == 'less_than':
            limit = ctx.get('lt')
            hint = f"{field}: the value must be less than {limit}."

        elif error_type in {'int_parsing', 'int_type', 'float_parsing', 'float_type'}:
            hint = f"{field}: provide a value of the correct numeric type."

        elif error_type == 'string_type':
            hint = f"{field}: provide a string."

        elif error_type == 'value_error':
            message = str(error.get('msg', 'Invalid value.'))
            message = message.removeprefix('Value error, ').rstrip('.')
            hint = f"{field}: {message}."

        else:
            hint = f"{field}: the value does not match the argument schema."

        hints.append(hint)

    return '; '.join(hints) or 'Correct the arguments according to the tool schema.'


def validation_error_result(message: str, hint: str) -> ToolResult:
    """
    Create a validation error result.

    Args:
        message (str): Error message describing the validation failure.
        hint (str): Instructions for correcting the input.

    Returns:
        ToolResult: Structured validation error result.
    """
    return ToolResult(
        status='error',
        error=ToolError(
            error_code=ToolErrorCode.VALIDATION,
            message=message,
            hint=hint,
            retryable=True,
        ),
    )


def unknown_tool_result(name: str, available: str) -> ToolResult:
    """
    Create an error result for an unknown tool.

    Args:
        name (str): Name of the requested tool.
        available (str): Comma-separated list of registered tool names.

    Returns:
        ToolResult: Structured unknown tool error result.
    """
    return ToolResult(
        status='error',
        error=ToolError(
            error_code=ToolErrorCode.UNKNOWN_TOOL,
            message=f"Tool '{name}' is not registered.",
            hint=(
                f'Choose a tool from the following list: {available}.'
                if available
                else 'There are no registered tools in the current request.'
            ),
            retryable=True,
        ),
    )


def execution_failed_result(tool_name: str) -> ToolResult:
    """
    Create a result for an internal tool execution failure.

    Args:
        tool_name (str): Name of the tool that failed.

    Returns:
        ToolResult: Structured execution failure result.
    """
    return ToolResult(
        status='error',
        error=ToolError(
            error_code=ToolErrorCode.EXECUTION_FAILED,
            message=f"Tool '{tool_name}' failed due to an internal error.",
            hint=(
                "The argument schema has already been validated. Do not guess at corrections "
                f"or retry the call to '{tool_name}'; choose another available approach."
            ),
            retryable=False,
        ),
    )


def temporary_failure_result(tool_name: str, retry_after: float | None) -> ToolResult:
    """
    Create a result for a temporary tool failure.
    Args:
        tool_name (str): Name of the temporarily unavailable tool.
        retry_after (float): Delay before retrying in seconds, if available.
    Returns:
        ToolResult: Structured temporary failure result.
    """
    if retry_after is None:
        hint = (
            f"Tool '{tool_name}' is temporarily unavailable. "
            "Do not retry immediately; choose another approach or try again later."
        )
    else:
        hint = (
            f"Tool '{tool_name}' is temporarily unavailable. Do not retry before {retry_after:.1f} seconds have passed; "
            "if waiting is not possible, choose another approach."
        )

    return ToolResult(
        status='error',
        error=ToolError(
            error_code=ToolErrorCode.TEMPORARY_FAILURE,
            message=f"Tool '{tool_name}' is temporarily unavailable.",
            hint=hint,
            retryable=True,
            retry_after=retry_after,
        ),
    )


def handle_execution_exception(tool_name: str, exc: Exception) -> ToolResult:
    """
    Classify an execution exception, log it, and return the appropriate result.

    Args:
        tool_name (str): Name of the tool that raised the exception.
        exc (Exception): Exception raised during tool execution.

    Returns:
        ToolResult: Structured temporary or internal execution error result.
    """
    if is_temporary_failure(exc):
        log_exception("Temporary tool failure.", tool_name, exc)
        return temporary_failure_result(
            tool_name,
            retry_after_seconds(exc),
        )

    log_exception("Tool execution failed.", tool_name, exc)

    return execution_failed_result(tool_name)
