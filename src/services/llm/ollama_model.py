import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ollama import AsyncClient

from .base_model import BaseLLMModel
from .schemas import Message, Reply, UnknownRoleError


class OllamaModel(BaseLLMModel):
    """
    Implement the chat model for the Ollama API using an asynchronous client."""

    def __init__(
            self,
            model_name: str,
            ollama_client: AsyncClient,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        """
        Initialize the Ollama chat model.

        Args:
            model_name (str): Ollama model name, for example, "llama3".
            ollama_client (AsyncClient): Pre-initialized asynchronous Ollama client.
            temperature (float): Sampling temperature for generation.
            max_tokens (int): Maximum number of tokens to predict.
        """
        super().__init__()
        self._model_name = model_name
        self._client = ollama_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
            stop: Sequence[str] | None = None,
    ) -> Reply:
        """
        Asynchronously send a chat request to Ollama and return a normalized Reply.

        Args:
            messages (Sequence[Mapping[str, Any]]): Chat messages.
            tools (tools: Sequence[dict[str, Any]] | None = None, optional): Tool schemas available to the model.
            timeout (float | None, optional): Maximum time to wait for the request.
            stop (Sequence[str] | None, optional): A sequence of strings to stop the response on.

        Returns:
            Reply: Normalized Ollama response including text, tool calls, token usage and cost.
        """
        allowed_roles = self.allowed_roles | {'tool'}

        for message in messages:
            role = message.get('role')

            if role not in allowed_roles:
                raise UnknownRoleError(
                    f"{self.__class__.__name__}: Unknown role '{role}'. Allowed roles: {sorted(allowed_roles)}")

        formatted_messages = []

        for message in messages:
            formatted_message = dict(message)

            tool_calls = formatted_message.get('tool_calls')

            if tool_calls:
                formatted_message['tool_calls'] = [
                    {
                        'type': 'function',
                        'function': {
                            'name': call['name'],
                            'arguments': call.get('arguments', {}),
                        },
                    }
                    for call in tool_calls
                ]

            if formatted_message.get('role') == 'tool':
                tool_name = formatted_message.get('tool_name') or formatted_message.get('name')

                if tool_name:
                    formatted_message['tool_name'] = tool_name

                formatted_message.pop('name', None)
                formatted_message.pop('tool_call_id', None)

            formatted_messages.append(formatted_message)

        options = {
            'num_predict': self._max_tokens,
            'temperature': self._temperature,
        }

        if stop:
            options['stop'] = list(stop)

        async def request() -> Any:
            """
            Execute an Ollama chat request.

            Returns:
                Any: Raw Ollama response.
            """
            return await self._client.chat(
                model=self._model_name,
                messages=formatted_messages,
                tools=list(tools) if tools else None,
                options=options,
            )

        if timeout is not None:
            response = await asyncio.wait_for(request(), timeout=timeout)
        else:
            response = await request()

        tool_calls = []

        for index, tool_call in enumerate(response.message.tool_calls or []):
            tool_calls.append({
                'id': f'{tool_call.function.name}:{index}',
                'name': tool_call.function.name,
                'arguments': dict(tool_call.function.arguments),
            })

        return Reply(
            text=response.message.content or '',
            tool_calls=tool_calls,
            model=self._model_name,
            in_tokens=response.prompt_eval_count or 0,
            out_tokens=response.eval_count or 0,
            cost_usd=0.0,
        )

    async def generate_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> str:
        """
        Asynchronously generate a complete text response via the Ollama API.

        Args:
            messages (Sequence[Message]): A sequence of messages representing the dialogue context.
            stop (Sequence[str] | None, optional): A sequence of strings to stop the response on.
            timeout (float | None, optional): Maximum time to wait for the response.

        Returns:
            str: The cleaned text response generated by Ollama.
        """
        # Roles validation
        self._validate_messages(messages)

        # Prepare messages
        formatted_messages = [
            {
                'role': message.role,
                'content': message.content,
            }
            for message in messages
        ]

        # Request
        options = {
            'num_predict': self._max_tokens,
            'temperature': self._temperature,
        }

        if stop:
            options['stop'] = list(stop)

        async def request() -> Any:
            """
            Execute an Ollama chat request.

            Returns:
                Any: Raw Ollama response.
            """
            return await self._client.chat(
                model=self._model_name,
                messages=formatted_messages,
                options=options,
            )

        if timeout is not None:
            response = await asyncio.wait_for(request(), timeout=timeout)
        else:
            response = await request()

        # Process response
        model_text = response.message.content

        return self._postprocess(model_text)

    async def get_streaming_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream the response from the Ollama API.

        Args:
            messages (Sequence[Message]): A sequence of messages representing the dialogue context.
            stop (Sequence[str] | None, optional): A sequence of strings to stop the response on.
            timeout (float | None, optional): Maximum time to wait for the response.

        Returns:
            AsyncIterator[str]: A stream of cleaned response text chunks.
        """
        # Roles validation
        self._validate_messages(messages)

        # Prepare messages
        formatted_messages = [
            {
                'role': message.role,
                'content': message.content,
            }
            for message in messages
        ]

        # Request
        options = {
            'num_predict': self._max_tokens,
            'temperature': self._temperature,
        }

        if stop:
            options['stop'] = list(stop)

        stream = await self._client.chat(
            model=self._model_name,
            messages=formatted_messages,
            stream=True,
            options=options,
        )

        # Process stream
        raw_text = ""
        emitted_text = ""

        if timeout is None:
            async for chunk in stream:
                content = chunk.message.content or ""
                if content:
                    raw_text += content
                    delta, emitted_text = self._stream_postprocess_delta(raw_text, emitted_text)
                    if delta:
                        yield delta
        else:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        anext(stream),
                        timeout=timeout,
                    )
                except StopAsyncIteration:
                    break

                content = chunk.message.content or ""
                if content:
                    raw_text += content
                    delta, emitted_text = self._stream_postprocess_delta(raw_text, emitted_text)
                    if delta:
                        yield delta
