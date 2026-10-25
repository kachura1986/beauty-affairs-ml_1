import hashlib
import inspect
import json
import logging
import math
import traceback
from collections.abc import Mapping
from typing import Any, Iterable

from pydantic import ValidationError

from .tool import Tool, ToolError, ToolErrorCode, ToolResult

MAX_TOOL_OUTPUT = 4000

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


def _is_temporary_failure(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    return any(cls.__name__ in _TEMPORARY_EXCEPTION_NAMES for cls in type(exc).__mro__)


def _retry_after_seconds(exc: Exception) -> float | None:
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


def _log_exception(message: str, tool_name: str, exc: Exception) -> None:
    logger.error(
        message,
        extra={
            'tool_name': tool_name,
            'exception_type': type(exc).__name__,
            'stack': traceback.format_tb(exc.__traceback__),
        },
    )


def _validation_hint(errors: list[dict[str, Any]]) -> str:
    hints = []

    for error in errors:
        location = '.'.join(str(x) for x in error.get('loc', ()))
        field = f"Поле '{location}'" if location else 'Аргументы'
        error_type = error.get('type', '')
        ctx = error.get('ctx') or {}

        if error_type == 'missing':
            hint = f"Укажи обязательное поле '{location}'."

        elif error_type == 'extra_forbidden':
            hint = f"Поле '{location}' не поддерживается схемой; удали его."

        elif error_type == 'string_too_short':
            limit = ctx.get('min_length')
            hint = f"{field}: укажи строку длиной не менее {limit} символов."

        elif error_type == 'string_too_long':
            limit = ctx.get('max_length')
            hint = f"{field}: сократи строку до {limit} символов."

        elif error_type == 'greater_than_equal':
            limit = ctx.get('ge')
            hint = f"{field}: значение должно быть не меньше {limit}."

        elif error_type == 'less_than_equal':
            limit = ctx.get('le')
            hint = f"{field}: значение должно быть не больше {limit}."

        elif error_type == 'greater_than':
            limit = ctx.get('gt')
            hint = f"{field}: значение должно быть больше {limit}."

        elif error_type == 'less_than':
            limit = ctx.get('lt')
            hint = f"{field}: значение должно быть меньше {limit}."

        elif error_type in {'int_parsing', 'int_type', 'float_parsing', 'float_type'}:
            hint = f"{field}: передай число правильного типа."

        elif error_type == 'string_type':
            hint = f"{field}: передай строку."

        elif error_type == 'value_error':
            message = str(error.get('msg', 'Некорректное значение.'))
            message = message.removeprefix('Value error, ').rstrip('.')

            if 'min_price must be less than or equal to max_price' in message:
                hint = 'Укажи min_price не больше max_price.'
            else:
                hint = f"{field}: {message}."

        else:
            hint = f"{field}: значение не соответствует схеме аргументов."

        hints.append(hint)

    return '; '.join(hints) or 'Исправь аргументы согласно схеме инструмента.'


class ToolRegistry:
    """Registry of tools available to the agent_graph."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        """
        Initialize the registry with the given tools.

        Args:
            tools: Iterable of Tool objects.
        """
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        """
        Return tool definitions in the format expected by the LLM.

        Returns:
            list[dict[str, Any]]: List of tool definitions.
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    @staticmethod
    def fingerprint(call: dict[str, Any]) -> str:
        """
        Return a stable fingerprint for a tool call.

        The fingerprint is based on the tool name and its normalized
        arguments. It is used to detect repeated identical calls.

        Args:
            call: Dictionary containing the tool name and arguments.

        Returns:
            str: Stable fingerprint for the tool call.
        """
        payload = json.dumps(
            {
                'n': call['name'],
                'a': call.get('arguments', {}),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @staticmethod
    def _validation_error(message: str, hint: str) -> ToolResult:
        return ToolResult(
            status='error',
            error=ToolError(
                error_code=ToolErrorCode.VALIDATION,
                message=message,
                hint=hint,
                retryable=True,
            ),
        )

    @staticmethod
    def _execution_failed_result(tool_name: str) -> ToolResult:
        return ToolResult(
            status='error',
            error=ToolError(
                error_code=ToolErrorCode.EXECUTION_FAILED,
                message=f"Инструмент '{tool_name}' завершился внутренней ошибкой.",
                hint=(
                    'Схема аргументов уже проверена. Не исправляй аргументы '
                    f"наугад и не повторяй вызов '{tool_name}'; "
                    'выбери другой доступный способ.'
                ),
                retryable=False,
            ),
        )

    @staticmethod
    def _temporary_failure_result(tool_name: str, retry_after: float | None) -> ToolResult:
        if retry_after is None:
            hint = (
                f"Инструмент '{tool_name}' временно недоступен. "
                'Не повторяй вызов немедленно; выбери другой способ или повтори позже.'
            )
        else:
            hint = (
                f"Инструмент '{tool_name}' временно недоступен. "
                f'Не повторяй раньше чем через {retry_after:.1f} с; '
                'если ожидание невозможно, выбери другой способ.'
            )

        return ToolResult(
            status='error',
            error=ToolError(
                error_code=ToolErrorCode.TEMPORARY_FAILURE,
                message=f"Инструмент '{tool_name}' временно недоступен.",
                hint=hint,
                retryable=True,
                retry_after=retry_after,
            ),
        )

    async def invoke(self, call: Any) -> ToolResult:
        """Execute a tool call and convert failures into ToolResult errors.

        This method does not propagate tool execution exceptions to the caller. Expected input errors and
        unexpected internal failures are represented as structured ToolError instances instead.

        Args:
            call: Dictionary containing the tool name and arguments.

        Returns:
            ToolResult: Result of the tool execution.
        """
        if not isinstance(call, Mapping):
            return self._validation_error(
                "Invalid tool call structure.",
                "Provide a mapping with 'name' and 'arguments' fields.",
            )

        name = call.get('name')
        if not isinstance(name, str) or not name.strip():
            return self._validation_error(
                "Invalid tool name.",
                "Provide a non-empty string in the 'name' field.",
            )

        if 'arguments' not in call:
            return self._validation_error(
                "Tool arguments are missing.",
                "Provide 'arguments' as an object, even if it is empty.",
            )

        args = call['arguments']
        if not isinstance(args, dict):
            return self._validation_error(
                "Invalid tool arguments structure.",
                "The 'arguments' field must be an object with named fields.",
            )

        tool = self._tools.get(name)

        # The model requested a tool that is not registered
        if tool is None:
            available = ', '.join(sorted(self._tools))

            return ToolResult(
                status='error',
                error=ToolError(
                    error_code=ToolErrorCode.UNKNOWN_TOOL,
                    message=f"Инструмент '{name}' не зарегистрирован.",
                    hint=(
                        f'Выбери инструмент из списка: {available}.'
                        if available
                        else 'В текущем запросе нет зарегистрированных инструментов.'
                    ),
                    retryable=True,
                ),
            )

        # Validate arguments before calling the tool
        try:
            validated = tool.args_model.model_validate(args)
            args = validated.model_dump()
        except ValidationError as exc:
            return self._validation_error('Invalid tool arguments.', _validation_hint(exc.errors()))

        try:
            out = tool.fn(**args)

            if inspect.isawaitable(out):
                out = await out

            out = str(out)

        except Exception as exc:  # noqa: BLE001
            if _is_temporary_failure(exc):
                _log_exception('Temporary tool failure.', name, exc)

                return self._temporary_failure_result(name, _retry_after_seconds(exc))

            _log_exception('Tool execution failed.', name, exc)

            return self._execution_failed_result(name)

        # Keep the model context bounded by limiting the tool output size
        truncated = len(out) > MAX_TOOL_OUTPUT

        return ToolResult(
            status='ok',
            content=out[:MAX_TOOL_OUTPUT],
            truncated=truncated,
        )
