import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from openai import AsyncAzureOpenAI

from .base_model import BaseLLMModel
from .schemas import Message, Reply, UnknownRoleError


class AzureOpenAIModel(BaseLLMModel):
    """
    Implement the chat model for the Azure OpenAI API using an asynchronous client.
    """

    def __init__(
            self,
            model_name: str,
            azure_openai_client: AsyncAzureOpenAI,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._client = azure_openai_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _extract_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []

        if not choices:
            return ""

        message = getattr(choices[0], "message", None)

        if message is None:
            return ""

        return getattr(message, "content", None) or ""

    def _format_chat_messages(
            self,
            messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        formatted_messages: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")

            if role not in self.allowed_chat_roles():
                raise UnknownRoleError(
                    f"{self.__class__.__name__}: Unknown role '{role}'. "
                    f"Allowed roles: {sorted(self.allowed_chat_roles())}"
                )

            formatted_message: dict[str, Any] = {
                "role": role,
                "content": message.get("content") or "",
            }

            name = message.get("name")

            if name:
                formatted_message["name"] = name

            if role == "assistant":
                raw_tool_calls = message.get("tool_calls") or []

                if raw_tool_calls:
                    tool_calls: list[dict[str, Any]] = []

                    for index, raw_tool_call in enumerate(raw_tool_calls):
                        if not isinstance(raw_tool_call, Mapping):
                            raise TypeError(
                                "Assistant tool call must be a mapping."
                            )

                        function = raw_tool_call.get("function")

                        if isinstance(function, Mapping):
                            function_data = dict(function)
                        else:
                            function_data = dict(raw_tool_call)

                        tool_name = (
                                function_data.get("name")
                                or raw_tool_call.get("name")
                        )

                        if not tool_name:
                            raise ValueError(
                                "Assistant tool call is missing function name."
                            )

                        arguments = (
                                function_data.get("arguments")
                                or function_data.get("args")
                                or raw_tool_call.get("arguments")
                                or raw_tool_call.get("args")
                                or {}
                        )

                        if not isinstance(arguments, str):
                            arguments = json.dumps(
                                arguments,
                                ensure_ascii=False,
                            )

                        tool_calls.append({
                            "id": str(
                                raw_tool_call.get("id")
                                or f"{tool_name}:{index}"
                            ),
                            "type": "function",
                            "function": {
                                "name": str(tool_name),
                                "arguments": arguments,
                            },
                        })

                    formatted_message["tool_calls"] = tool_calls

            elif role == "tool":
                tool_name = (
                        message.get("tool_name")
                        or message.get("name")
                )
                tool_call_id = message.get("tool_call_id")

                if tool_name:
                    formatted_message["name"] = tool_name

                if tool_call_id:
                    formatted_message["tool_call_id"] = str(tool_call_id)

            formatted_messages.append(formatted_message)

        return formatted_messages

    def allowed_chat_roles(self) -> set[str]:
        return self.allowed_roles | {"tool"}

    @staticmethod
    def _extract_tool_calls(
            response: Any,
    ) -> list[dict[str, Any]]:
        choices = getattr(response, "choices", None) or []

        if not choices:
            return []

        message = getattr(choices[0], "message", None)

        if message is None:
            return []

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        result: list[dict[str, Any]] = []

        for index, tool_call in enumerate(raw_tool_calls):
            function = getattr(tool_call, "function", None)

            if function is None:
                continue

            name = getattr(function, "name", None)

            if not name:
                continue

            raw_arguments = getattr(function, "arguments", None) or "{}"

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON arguments for tool '{name}'."
                    ) from exc
            else:
                arguments = raw_arguments

            if not isinstance(arguments, Mapping):
                raise TypeError(
                    f"Arguments for tool '{name}' must be a mapping."
                )

            call_id = getattr(tool_call, "id", None)

            result.append({
                "id": str(call_id or f"{name}:{index}"),
                "name": str(name),
                "arguments": dict(arguments),
            })

        return result

    @staticmethod
    def _extract_usage(
            response: Any,
    ) -> tuple[int, int]:
        usage = getattr(response, "usage", None)

        if usage is None:
            return 0, 0

        input_tokens = getattr(usage, "prompt_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or 0

        return int(input_tokens), int(output_tokens)

    async def generate_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> str:
        self._validate_messages(messages)

        for message in messages:
            if message.role == "tool":
                raise ValueError(
                    "generate_response() does not support tool messages. "
                    "Use chat() for tool calling."
                )

        formatted_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        async def request() -> Any:
            return await self._client.chat.completions.create(
                model=self._model_name,
                messages=formatted_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=list(stop) if stop else None,
            )

        if timeout is not None:
            response = await asyncio.wait_for(
                request(),
                timeout=timeout,
            )
        else:
            response = await request()

        text = self._extract_text(response)

        return self._postprocess(text)

    async def get_streaming_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> AsyncIterator[str]:
        self._validate_messages(messages)

        for message in messages:
            if message.role == "tool":
                raise ValueError(
                    "get_streaming_response() does not support tool messages. "
                    "Use chat() for tool calling."
                )

        formatted_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        async def request() -> Any:
            return await self._client.chat.completions.create(
                model=self._model_name,
                messages=formatted_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=list(stop) if stop else None,
                stream=True,
            )

        stream = await request()

        raw_text = ""
        emitted_text = ""

        try:
            while True:
                if timeout is None:
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                else:
                    try:
                        chunk = await asyncio.wait_for(
                            anext(stream),
                            timeout=timeout,
                        )
                    except StopAsyncIteration:
                        break

                choices = getattr(chunk, "choices", None) or []

                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)

                if delta is None:
                    continue

                chunk_text = getattr(delta, "content", None)

                if not chunk_text:
                    continue

                raw_text += chunk_text

                text_delta, emitted_text = self._stream_postprocess_delta(
                    raw_text,
                    emitted_text,
                )

                if text_delta:
                    yield text_delta

        finally:
            close = getattr(stream, "close", None)

            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
            stop: Sequence[str] | None = None,
    ) -> Reply:
        formatted_messages = self._format_chat_messages(messages)

        async def request() -> Any:
            return await self._client.chat.completions.create(
                model=self._model_name,
                messages=formatted_messages,
                tools=list(tools) if tools else None,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=list(stop) if stop else None,
            )

        if timeout is not None:
            response = await asyncio.wait_for(
                request(),
                timeout=timeout,
            )
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
