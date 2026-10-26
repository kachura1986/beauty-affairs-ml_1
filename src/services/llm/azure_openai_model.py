import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI

from .base_model import BaseLLMModel
from .schemas import Reply


class AzureOpenAIModel(BaseLLMModel):
    """Implement the chat model for the Azure OpenAI API using an asynchronous client."""

    def __init__(
            self,
            model_name: str,
            azure_openai_client: AsyncOpenAI,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        """
        Initialize the Azure OpenAI chat model.

        Args:
            model_name (str): Azure OpenAI deployment or model name.
            azure_openai_client (AsyncOpenAI): Pre-initialized asynchronous Azure OpenAI client.
            temperature (float): Sampling temperature for generation.
            max_tokens (int): Maximum number of tokens to generate.
        """
        super().__init__()

        self._model_name = model_name
        self._client = azure_openai_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def _format_chat_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate and format chat messages for the Azure OpenAI API.

        Args:
            messages (Sequence[Mapping[str, Any]]): Messages to format, including optional tool calls.

        Returns:
            list[dict[str, Any]]: Messages in the format expected by the API.

        Raises:
            UnknownRoleError: If a message contains an unsupported role.
            TypeError: If an assistant tool call is not a mapping.
            ValueError: If a tool call has no function name.
        """
        # Validate role
        self._validate_roles(messages)

        formatted_messages: list[dict[str, Any]] = []

        for message in messages:
            role = message.get('role')

            content = message.get('content') or ''

            if role == 'tool':
                tool_call_id = message.get('tool_call_id')

                if not tool_call_id:
                    raise ValueError(f"{self.__class__.__name__}: Tool message is missing tool_call_id.")

                formatted_messages.append({
                    'type': 'function_call_output',
                    'call_id': str(tool_call_id),
                    'output': content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                })
                continue

            if role == 'assistant':
                raw_tool_calls = message.get('tool_calls') or []

                if content or not raw_tool_calls:
                    formatted_messages.append({
                        'role': role,
                        'content': content,
                    })

                for index, raw_tool_call in enumerate(raw_tool_calls):
                    if not isinstance(raw_tool_call, Mapping):
                        raise TypeError(f"{self.__class__.__name__}: Assistant tool call must be a mapping.")

                    function = raw_tool_call.get('function')

                    if isinstance(function, Mapping):
                        function_data = dict(function)
                    else:
                        function_data = dict(raw_tool_call)

                    tool_name = function_data.get('name') or raw_tool_call.get('name')

                    if not tool_name:
                        raise ValueError(f"{self.__class__.__name__}: Assistant tool call is missing function name.")

                    arguments = (
                            function_data.get('arguments')
                            or function_data.get('args')
                            or raw_tool_call.get('arguments')
                            or raw_tool_call.get('args')
                            or {}
                    )

                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)

                    formatted_messages.append({
                        'type': 'function_call',
                        'call_id': str(raw_tool_call.get('id') or f'{tool_name}:{index}'),
                        'name': str(tool_name),
                        'arguments': arguments,
                    })

            else:
                formatted_messages.append({
                    'role': role,
                    'content': content,
                })

        return formatted_messages

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Extract response text from the first completion choice.

        Args:
            response (Any): Raw Azure OpenAI completion response.

        Returns:
            str: Response text, or an empty string if unavailable.
        """
        return getattr(response, 'output_text', None) or ''

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """
        Extract and normalize tool calls from a completion response.

        Args:
            response (Any): Raw Azure OpenAI completion response.

        Returns:
            list[dict[str, Any]]: Tool calls with IDs, function names, and argument mappings.

        Raises:
            ValueError: If tool-call arguments contain invalid JSON.
            TypeError: If tool-call arguments are not a mapping.
        """
        raw_output = getattr(response, 'output', None) or []
        result: list[dict[str, Any]] = []

        for index, tool_call in enumerate(raw_output):
            if getattr(tool_call, 'type', None) != 'function_call':
                continue

            name = getattr(tool_call, 'name', None)

            if not name:
                continue

            raw_arguments = getattr(tool_call, 'arguments', None) or '{}'

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.__class__.__name__}: Invalid JSON arguments for tool '{name}'.") from exc
            else:
                arguments = raw_arguments

            if not isinstance(arguments, Mapping):
                raise TypeError(f"{self.__class__.__name__}: Arguments for tool '{name}' must be a mapping.")

            call_id = getattr(tool_call, 'call_id', None)

            result.append({
                'id': str(call_id or f'{name}:{index}'),
                'name': str(name),
                'arguments': dict(arguments),
            })

        return result

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int, int]:
        """
        Extract input and output token counts from a completion response.

        Args:
            response (Any): Raw Azure OpenAI completion response.

        Returns:
            tuple[int, int]: Input token count and output token count, defaulting to zero.
        """
        usage = getattr(response, 'usage', None)

        if usage is None:
            return 0, 0

        input_tokens = getattr(usage, 'input_tokens', None) or 0
        output_tokens = getattr(usage, 'output_tokens', None) or 0

        return int(input_tokens), int(output_tokens)

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
    ) -> Reply:
        """
        Send a chat request and return a normalized response.

        Args:
            messages (Sequence[Mapping[str, Any]]): Chat messages, including optional tool-call data.
            tools (Sequence[dict[str, Any]] | None, optional): Tool schemas available to the model.
            timeout (float | None, optional): Maximum time to wait for the request.

        Returns:
            Reply: Normalized response containing text, tool calls, token usage, and cost.
        """
        formatted_messages = self._format_chat_messages(messages)

        formatted_tools = None

        if tools:
            formatted_tools = []

            for tool in tools:
                formatted_tool = dict(tool)
                function = formatted_tool.get('function')

                if isinstance(function, Mapping):
                    formatted_tool.pop('function')
                    formatted_tool.update(dict(function))

                formatted_tools.append(formatted_tool)

        async def request() -> Any:
            """Execute an Azure OpenAI completion request.

            Returns:
                Any: Raw Azure OpenAI completion response.
            """
            return await self._client.responses.create(
                model=self._model_name,
                input=formatted_messages,
                tools=formatted_tools,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )

        if timeout is not None:
            response = await asyncio.wait_for(request(), timeout=timeout)
        else:
            response = await request()

        text = self._extract_text(response)
        tool_calls = self._extract_tool_calls(response)
        input_tokens, output_tokens = self._extract_usage(response)

        return Reply(
            text=self._postprocess(text),
            tool_calls=tool_calls,
            model=self._model_name,
            in_tokens=input_tokens,
            out_tokens=output_tokens,
            cost_usd=0.0,
        )
