import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any, Iterable

from pydantic import ValidationError

from .errors import handle_execution_exception, unknown_tool_result, validation_error_result, validation_hint
from .tool import Tool, ToolResult, ToolStatus

MAX_TOOL_OUTPUT = 4000


class ToolRegistry:
    """Registry of tools available to the agent_graph."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        """
        Initialize the registry with the given tools.

        Args:
            tools: Iterable of Tool objects.
        """
        self._tools = {}

        for tool in tools:
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

            if inspect.isawaitable(out):
                out = await out

            out = str(out)

        except Exception as exc:  # noqa: BLE001
            return handle_execution_exception(name, exc)

        # Keep the model context bounded by limiting the tool output size
        truncated = len(out) > MAX_TOOL_OUTPUT

        return ToolResult(
            status=ToolStatus.OK,
            content=out[:MAX_TOOL_OUTPUT],
            truncated=truncated,
        )
