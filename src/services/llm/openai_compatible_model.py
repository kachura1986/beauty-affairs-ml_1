import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from .base_model import BaseLLMModel
from .schemas import Reply, ToolCall


def _is_transient_openai_error(exc: Exception) -> bool:
    """Classify an OpenAI-SDK-path exception as transient (retryable) or permanent.

    CONTRACT: single source of truth for the OpenAI path; ``LLMModel.is_transient_error`` delegates here.
    ``APITimeoutError`` / ``httpx.TimeoutException`` are permanent (billed SDK call; a retry would double-bill) —
    contractual only today, because the SDK timeout is disabled (``timeout=None`` + ``asyncio.wait_for``).
    Builtin ``TimeoutError`` from ``asyncio.wait_for`` is the production timeout path and is transient.

    STATUS CODE RESOLUTION: ``openai.APIStatusError`` (and subclasses) exposes ``status_code`` as a direct
    attribute; a bare ``httpx.HTTPStatusError`` keeps it on ``exc.response.status_code``. Both paths are
    checked so a 429/5xx is never silently downgraded to permanent, regardless of which layer wraps the
    response today; other statuses (e.g. 4xx except 429) intentionally fall through to permanent.
    ``httpx.HTTPStatusError`` is not a ``httpx.TransportError``, so step 3 does not cover it.

    Args:
        exc (Exception): Exception raised during ``OpenAICompatibleModel.chat``.

    Returns:
        bool: True if a retry has a non-trivial chance of succeeding.
    """
    # Step 1: SDK-level timeout → permanent (billed; retry would double-bill).
    # Contractual today: SDK timeout is disabled; kept for the day it is re-enabled.
    if isinstance(exc, APITimeoutError):
        return False
    if isinstance(exc.__cause__, httpx.TimeoutException):
        return False

    # Step 2: asyncio.wait_for timeout → transient (budget may still have room).
    # Order after step 1 is deliberate: APITimeoutError is not a TimeoutError subclass.
    if isinstance(exc, TimeoutError):
        return True

    # Step 3: connection-level errors → transient.
    if isinstance(exc, (ConnectionError, APIConnectionError)):
        return True
    if isinstance(exc.__cause__, (ConnectionError, httpx.TransportError)):
        return True

    # Step 4: HTTP 429 / 5xx → transient.
    # SDK-path: `openai.APIStatusError` (and subclasses) exposes `status_code` directly.
    # Bare-httpx path: `httpx.HTTPStatusError` keeps it on `exc.response.status_code`.
    status = getattr(exc, 'status_code', None)
    if status is None:
        response = getattr(exc, 'response', None)
        status = getattr(response, 'status_code', None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True

    # Fallthrough: unclassified errors are permanent on purpose — retrying an already-billed call would double-charge.
    # Only steps 1–4 carry a clear retry signal; provider SDK wrappers without a builtin/httpx cause don't.
    return False


class OpenAICompatibleModel(BaseLLMModel):
    """Implement the chat model for any OpenAI-compatible Responses API endpoint over an asynchronous client."""

    def __init__(
            self,
            model_name: str,
            client: AsyncOpenAI,
            temperature: float,
            max_tokens: int,
            input_cost_per_million: float,
            output_cost_per_million: float,
    ) -> None:
        """
        Initialize the OpenAI-compatible chat model.

        Args:
            model_name (str): Provider-specific deployment or model name (e.g. Azure deployment name, Ollama model tag).
            client (AsyncOpenAI): Pre-initialized asynchronous OpenAI-compatible client.
            temperature (float): Sampling temperature for generation.
            max_tokens (int): Maximum number of tokens to generate.
            input_cost_per_million (float): Cost per 1M input tokens, in the currency of the configured tariff.
            output_cost_per_million (float): Cost per 1M output tokens, in the currency of the configured tariff.
        """
        super().__init__()

        self._model_name = model_name
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._input_cost_per_million = input_cost_per_million
        self._output_cost_per_million = output_cost_per_million

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def _format_chat_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate and format chat messages for the OpenAI-compatible API.

        CONTRACT: ``role='system'`` messages are dropped — the system prompt goes via ``chat(..., system_prompt=...)``
        (``instructions``), never through ``input``. Tool calls without an explicit ``id`` get a deterministic
        synthetic id ``f'{name}:{index}'`` so the tool result can be paired back.

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

            # Skip messages with 'system' role, as they are supplied separately to the input
            if role == 'system':
                continue

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

                    tool_name = raw_tool_call.get('name')

                    if not tool_name:
                        raise ValueError(f"{self.__class__.__name__}: Assistant tool call is missing function name.")

                    arguments = raw_tool_call.get('arguments') or {}

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
        Extract text from a Responses API response.

        Args:
            response (Any): Raw OpenAI-compatible Responses API response.

        Returns:
            str: Response text, or an empty string if unavailable.
        """
        return getattr(response, 'output_text', None) or ''

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        """
        Extract and normalize tool calls from a Responses API response.

        CONTRACT: non-``function_call`` output items are silently skipped. Missing ``call_id`` gets
        a deterministic synthetic id ``f'{name}:{index}'`` so the tool result can be paired back.

        Args:
            response (Any): Raw OpenAI-compatible Responses API response.

        Returns:
            list[ToolCall]: Tool calls with IDs, function names, and argument mappings.

        Raises:
            ValueError: If tool-call arguments contain invalid JSON.
            TypeError: If tool-call arguments are not a mapping.
        """
        raw_output = getattr(response, 'output', None) or []
        result: list[ToolCall] = []

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
            response (Any): Raw OpenAI-compatible completion response.

        Returns:
            tuple[int, int]: Input token count and output token count, defaulting to zero.
        """
        usage = getattr(response, 'usage', None)

        if usage is None:
            return 0, 0

        input_tokens = getattr(usage, 'input_tokens', None) or 0
        output_tokens = getattr(usage, 'output_tokens', None) or 0

        return int(input_tokens), int(output_tokens)

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Compute the request cost in currency from token counts and configured per-million rates.

        CONTRACT: returns ``0.0`` when both token counts are ``<= 0`` — cost-based budget limits are not enforced for
        that call. The caller must not treat ``0.0`` as "free call", only as "no accounting data".

        Args:
            input_tokens (int): Number of input tokens reported by the API.
            output_tokens (int): Number of output tokens reported by the API.

        Returns:
            float: Cost in the currency of the configured tariff. Zero if both rates are explicitly zero or token stats
                are missing (in which case cost-based budget limits are not enforced for this call).
        """
        input_rate = self._input_cost_per_million
        output_rate = self._output_cost_per_million

        if input_tokens <= 0 and output_tokens <= 0:
            return 0.0

        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
            system_prompt: str | None = None,
    ) -> Reply:
        """
        Send a chat request and return a normalized response.

        CONTRACT: ``timeout`` is required — the caller (``LLMStepService`` / ``FinalizeService``) owns the per-call
        wall-clock cap; SDK-level timeouts are disabled in the composition root (``timeout=None``) so that
        ``asyncio.wait_for`` here is the single source of timeout enforcement.

        Args:
            messages (Sequence[Mapping[str, Any]]): Chat messages, including optional tool-call data.
            tools (Sequence[dict[str, Any]] | None, optional): Tool schemas available to the model.
            timeout (float | None, optional): Maximum time to wait for the request.
            system_prompt (str | None, optional): System instructions.

        Returns:
            Reply: Normalized response containing text, tool calls, token usage, and cost
                (in the currency of the configured tariff).
        """
        if system_prompt is not None:
            system_prompt = system_prompt.strip() or None

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

        if timeout is None:
            raise ValueError(f"{self.__class__.__name__}: timeout is required.")

        request = self._client.responses.create(
            model=self._model_name,
            input=formatted_messages,
            tools=formatted_tools,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            instructions=system_prompt or None,
        )

        # asyncio.wait_for raises builtin TimeoutError on expiry, not openai.APITimeoutError.
        # _is_transient_openai_error treats that builtin TimeoutError as transient.
        response = await asyncio.wait_for(request, timeout=timeout)

        text = self._extract_text(response)
        tool_calls = self._extract_tool_calls(response)
        input_tokens, output_tokens = self._extract_usage(response)
        cost = self._compute_cost(input_tokens, output_tokens)

        return Reply(
            text=self._postprocess(text),
            tool_calls=tool_calls,
            model=self._model_name,
            in_tokens=input_tokens,
            out_tokens=output_tokens,
            cost=cost,
        )

    def is_transient_error(self, exc: Exception) -> bool:
        """Delegate to the module-level OpenAI classifier; SDK knowledge stays here."""
        return _is_transient_openai_error(exc)
