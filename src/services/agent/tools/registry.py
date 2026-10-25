import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any, Iterable

from pydantic import ValidationError

from .tool import Tool, ToolError, ToolErrorCode, ToolResult

MAX_TOOL_OUTPUT = 4000


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

    async def invoke(self, call: Any) -> ToolResult:
        """Execute a tool call and convert failures into ToolResult errors.

        This method does not propagate tool execution exceptions to the  caller. Expected input errors and
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
            return ToolResult(
                status='error',
                error=ToolError(
                    error_code=ToolErrorCode.UNKNOWN_TOOL,
                    message=f'Tool {name!r} is not registered.',
                    hint='Use one of the available tools: ' + ', '.join(sorted(self._tools)),
                    retryable=True,
                ),
            )

        # Validate arguments before calling the tool
        try:
            validated = tool.args_model.model_validate(args)
            args = validated.model_dump()
        except ValidationError as exc:
            errors = []

            for error in exc.errors():
                location = '.'.join(str(x) for x in error['loc'])
                errors.append(f"{location}: {error['msg']}")

            return self._validation_error(
                'Invalid tool arguments.',
                '; '.join(errors),
            )

        try:
            out = tool.fn(**args)

            if inspect.isawaitable(out):
                out = await out

            out = str(out)

        # TypeError usually means that the provided arguments do not match the tool function signature
        except TypeError:
            return ToolResult(
                status='error',
                error=ToolError(
                    error_code=ToolErrorCode.VALIDATION,
                    message='Tool arguments have an invalid format.',
                    hint='Check the argument types and values.',
                    retryable=True,
                ),
            )

        # ValueError is treated as a validation error raised by the tool
        # itself when the provided values are semantically invalid
        except ValueError as exc:
            return ToolResult(
                status='error',
                error=ToolError(
                    error_code=ToolErrorCode.VALIDATION,
                    message=str(exc),
                    hint='Fix the arguments according to the tool description.',
                    retryable=True,
                ),
            )

        # Any other exception is considered an internal tool failure
        # Do not expose implementation details or tracebacks to the model
        except Exception:
            return ToolResult(
                status='error',
                error=ToolError(
                    error_code=ToolErrorCode.INVARIANT_VIOLATED,
                    message='The tool failed with an internal error.',
                    hint='Do not repeat this call; choose another way to perform the action.',
                    retryable=False,
                ),
            )

        # Keep the model context bounded by limiting the tool output size
        truncated = len(out) > MAX_TOOL_OUTPUT

        return ToolResult(
            status='ok',
            content=out[:MAX_TOOL_OUTPUT],
            truncated=truncated,
        )
