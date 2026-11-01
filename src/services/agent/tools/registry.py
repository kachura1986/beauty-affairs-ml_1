import asyncio
import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any, Iterable

from pydantic import ValidationError

from services.llm import ToolCall

from .errors import handle_execution_exception, unknown_tool_result, validation_error_result, validation_hint
from .tool import Tool, ToolResult, ToolStatus
from .truncation import truncate_for_model


def _is_async_callable(fn: Any) -> bool:
    """
    Return True only if fn is an ``async def`` callable.

    Covers plain ``async def`` functions, ``functools.partial`` wrapping an ``async def`` (inspect handles it),
    and objects whose ``__call__`` is ``async def``.

    Deliberately rejects regular functions that merely *return* an awaitable: their synchronous prologue runs on
    the event loop *before* ``asyncio.wait_for`` sees the coroutine, so the timeout and budget guarantees would not hold.

    Args:
        fn: Callable to check.

    Returns:
        bool: True if fn is an async callable, False otherwise.
    """
    if inspect.iscoroutinefunction(fn):
        return True

    call = getattr(fn, '__call__', None)

    return call is not None and inspect.iscoroutinefunction(call)


class ToolRegistry:
    """Registry of tools available to the agent_graph."""

    def __init__(self, tools: Iterable[Tool], default_timeout_seconds: float, max_output_chars: int) -> None:
        """
        Initialize the registry with the given tools.

        Args:
            tools: Iterable of Tool objects.
            default_timeout_seconds: Fallback timeout applied to tools that do not define ``timeout_seconds``.
            max_output_chars: Fallback output character budget applied to tools that do not define ``max_output_chars``.
        """
        self._default_timeout_seconds = default_timeout_seconds
        self._max_output_chars = max_output_chars
        self._tools = {}

        for tool in tools:
            # Async-only contract: a sync callable would bypass asyncio.wait_for below,
            # and with it both the per-tool timeout and the run budget.
            if not _is_async_callable(tool.fn):
                raise TypeError(
                    f"{self.__class__.__name__}: tool '{tool.name}' must be an async callable; "
                    f"got {type(tool.fn).__name__}."
                )

            # Check for duplicate tool names
            if tool.name in self._tools:
                raise ValueError(f"{self.__class__.__name__}: Duplicate tool name: '{tool.name}'.")

            self._tools[tool.name] = tool

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
    def fingerprint(call: ToolCall) -> str:
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

    async def invoke(self, call: Any, *, timeout: float | None = None, truncate: bool | None = None) -> ToolResult:
        """Execute a tool call and convert failures into ToolResult errors.

        This method does not propagate tool execution exceptions to the caller. Expected input errors and
        unexpected internal failures are represented as structured ToolError instances instead.

        Args:
            call: Dictionary containing the tool name and arguments.
            timeout: Optional timeout in seconds for the tool execution.
            truncate: Optional flag to truncate the output.

        Returns:
            ToolResult: Result of the tool execution.
        """
        if not isinstance(call, Mapping):
            return validation_error_result(
                "Invalid tool call structure.",
                "Provide a mapping with 'name' and 'arguments' fields.",
            )

        name = call.get('name')
        if not isinstance(name, str) or not name.strip():
            return validation_error_result(
                "Invalid tool name.",
                "Provide a non-empty string in the 'name' field.",
            )

        if 'arguments' not in call:
            return validation_error_result(
                "Tool arguments are missing.",
                "Provide 'arguments' as an object, even if it is empty.",
            )

        args = call['arguments']
        if not isinstance(args, dict):
            return validation_error_result(
                "Invalid tool arguments structure.",
                "The 'arguments' field must be an object with named fields.",
            )

        tool = self._tools.get(name)

        # The model requested a tool that is not registered
        if tool is None:
            available = ', '.join(sorted(self._tools))

            return unknown_tool_result(name, available)

        # Validate arguments before calling the tool
        try:
            validated = tool.args_model.model_validate(args)
            args = validated.model_dump()
        except ValidationError as exc:
            return validation_error_result(
                'Invalid tool arguments.',
                validation_hint(exc.errors()),
            )

        try:
            out = tool.fn(**args)
        except Exception as exc:  # noqa: BLE001
            return handle_execution_exception(name, exc)

        # Registry guarantees fn is async (see __init__). This is an invariant check, not a fallback:
        # if it ever fires, the per-tool timeout and run budget would be silently bypassed — fail loud instead.
        if not inspect.isawaitable(out):
            raise TypeError(f"Tool '{name}' returned a non-awaitable; async callable expected.")

        per_tool_cap = tool.timeout_seconds if tool.timeout_seconds is not None else self._default_timeout_seconds
        # Both bounds apply: caller's `timeout` (typically budget.remaining_seconds()) and the per-tool ceiling.
        # The per-tool ceiling is authoritative and dominates the caller when it is smaller: a tool declared with
        # timeout_seconds=0.05 must not run for 300s just because the run budget allowed it.
        effective = min(per_tool_cap, timeout) if timeout is not None else per_tool_cap

        try:
            out = await asyncio.wait_for(out, timeout=effective)
        except Exception as exc:  # noqa: BLE001
            return handle_execution_exception(name, exc)

        do_truncate = tool.truncate if truncate is None else truncate
        budget = tool.max_output_chars if tool.max_output_chars is not None else self._max_output_chars

        content, was_truncated = truncate_for_model(out, budget if do_truncate else None)

        return ToolResult(
            status=ToolStatus.OK,
            content=content,
            truncated=was_truncated,
        )
