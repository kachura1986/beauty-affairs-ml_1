import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types

from .base_model import BaseLLMModel
from .schemas import CANONICAL_ROLES, Message, Reply, UnknownRoleError


class GeminiModel(BaseLLMModel):
    """
    Gemini implementation based on the modern google-genai SDK.

    Supported:
        - system messages
        - user messages
        - assistant/model messages
        - tool messages
        - normal generation
        - streaming generation
        - manual function calling
        - thought signatures
        - token usage
        - request timeout
    """

    def __init__(
            self,
            model_name: str,
            genai_client: genai.Client,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        super().__init__()

        self._model_name = model_name
        self._client = genai_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    # -------------------------------------------------------------------------
    # TEXT
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Extract text from Gemini response.

        Function-calling responses can have no text.
        """
        try:
            text = response.text
        except (AttributeError, ValueError):
            text = None

        if text:
            return text

        candidates = getattr(response, "candidates", None) or []

        if not candidates:
            return ""

        content = getattr(candidates[0], "content", None)

        if content is None:
            return ""

        parts = getattr(content, "parts", None) or []

        text_parts: list[str] = []

        for part in parts:
            part_text = getattr(part, "text", None)

            if part_text:
                text_parts.append(part_text)

        return "".join(text_parts)

    # -------------------------------------------------------------------------
    # TOOL SCHEMAS
    # -------------------------------------------------------------------------

    @classmethod
    def _build_function_declaration(
            cls,
            function: Mapping[str, Any],
    ) -> types.FunctionDeclaration:
        """
        Convert a generic/OpenAI-style function schema into
        google-genai FunctionDeclaration.

        Expected:

        {
            "name": "my_function",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        The SDK supports parameters_json_schema directly.
        """

        data = dict(function)

        data.pop("type", None)

        if "parameters" in data and "parameters_json_schema" not in data:
            data["parameters_json_schema"] = data.pop("parameters")

        if "response" in data and "response_json_schema" not in data:
            # Keep an explicitly provided response field as-is.
            pass

        name = data.get("name")

        if not name:
            raise ValueError(
                "Gemini function declaration is missing 'name'."
            )

        return types.FunctionDeclaration(**data)

    @classmethod
    def _build_gemini_tools(
            cls,
            tools: Sequence[dict[str, Any]] | None,
    ) -> list[types.Tool] | None:
        """
        Convert generic/OpenAI-style tool schemas into Gemini Tool objects.

        Supported input formats:

        OpenAI-style:

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Direct function:

        {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }

        Gemini-style:

        {
            "function_declarations": [
                {
                    "name": "...",
                    "description": "...",
                    "parameters_json_schema": {...}
                }
            ]
        }
        """

        if not tools:
            return None

        result: list[types.Tool] = []

        for tool in tools:
            if isinstance(tool, types.Tool):
                result.append(tool)
                continue

            if not isinstance(tool, Mapping):
                raise TypeError(
                    "Gemini tool must be a mapping or types.Tool, "
                    f"got {type(tool).__name__}."
                )

            # -----------------------------------------------------------------
            # Gemini-style Tool
            # -----------------------------------------------------------------

            function_declarations = tool.get("function_declarations")

            if function_declarations is not None:
                declarations: list[types.FunctionDeclaration] = []

                for declaration in function_declarations:
                    if isinstance(
                            declaration,
                            types.FunctionDeclaration,
                    ):
                        declarations.append(declaration)

                    elif isinstance(declaration, Mapping):
                        declarations.append(
                            types.FunctionDeclaration(
                                **dict(declaration)
                            )
                        )

                    else:
                        raise TypeError(
                            "Each function declaration must be either "
                            "a mapping or types.FunctionDeclaration."
                        )

                if declarations:
                    result.append(
                        types.Tool(
                            function_declarations=declarations
                        )
                    )

                continue

            # -----------------------------------------------------------------
            # OpenAI-style tool wrapper
            # -----------------------------------------------------------------

            function = tool.get("function")

            if isinstance(function, Mapping):
                declaration_data = dict(function)
            else:
                # Direct function declaration.
                declaration_data = dict(tool)

            declaration_data.pop("type", None)

            declaration = cls._build_function_declaration(
                declaration_data
            )

            result.append(
                types.Tool(
                    function_declarations=[declaration]
                )
            )

        return result or None

    # -------------------------------------------------------------------------
    # TOOL RESULTS
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_result(
            value: Any,
    ) -> dict[str, Any]:
        """
        Gemini FunctionResponse.response must be a JSON object.

        If a tool returns a mapping, use it directly.

        If it returns a scalar/string, wrap it into "output".

        A JSON string containing an object is decoded first.
        """

        if isinstance(value, Mapping):
            return dict(value)

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"output": value}

            if isinstance(parsed, Mapping):
                return dict(parsed)

            return {"output": parsed}

        return {"output": value}

    # -------------------------------------------------------------------------
    # MESSAGE CONVERSION
    # -------------------------------------------------------------------------

    @classmethod
    def _messages_to_contents(
            cls,
            messages: Sequence[Mapping[str, Any]],
    ) -> tuple[list[types.Content], str | None]:
        """
        Convert generic messages into Gemini Content objects.

        Mapping:

            system    -> GenerateContentConfig.system_instruction
            user      -> Content(role="user")
            assistant -> Content(role="model")
            tool      -> Content(role="tool")
        """

        contents: list[types.Content] = []

        system_messages: list[str] = []

        # Used when a tool result has tool_call_id but no tool name.
        tool_call_names: dict[str, str] = {}

        for message in messages:
            role = message.get("role")

            if role not in CANONICAL_ROLES:
                raise UnknownRoleError(
                    f"{cls.__name__}: Unknown role '{role}'. "
                    f"Allowed roles: {sorted(CANONICAL_ROLES)}"
                )

            # -----------------------------------------------------------------
            # SYSTEM
            # -----------------------------------------------------------------

            if role == "system":
                content = message.get("content")

                if content:
                    system_messages.append(str(content))

                continue

            # -----------------------------------------------------------------
            # USER
            # -----------------------------------------------------------------

            if role == "user":
                text = message.get("content") or ""

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=str(text)
                            )
                        ],
                    )
                )

                continue

            # -----------------------------------------------------------------
            # ASSISTANT
            # -----------------------------------------------------------------

            if role == "assistant":
                parts: list[types.Part] = []

                text = message.get("content") or ""

                if text:
                    parts.append(
                        types.Part.from_text(
                            text=str(text)
                        )
                    )

                raw_tool_calls = message.get("tool_calls") or []

                for raw_tool_call in raw_tool_calls:
                    if not isinstance(raw_tool_call, Mapping):
                        raise TypeError(
                            "Assistant tool call must be a mapping."
                        )

                    function = raw_tool_call.get("function")

                    if isinstance(function, Mapping):
                        tool_name = function.get("name")

                        arguments = (
                            function.get("arguments")
                            or function.get("args")
                            or {}
                        )
                    else:
                        tool_name = raw_tool_call.get("name")

                        arguments = (
                            raw_tool_call.get("arguments")
                            or raw_tool_call.get("args")
                            or {}
                        )

                    if not tool_name:
                        raise ValueError(
                            "Assistant tool call is missing function name."
                        )

                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON arguments for "
                                f"tool '{tool_name}'."
                            ) from exc

                    if not isinstance(arguments, Mapping):
                        raise TypeError(
                            f"Arguments for tool '{tool_name}' must be "
                            f"a mapping."
                        )

                    tool_call_id = raw_tool_call.get("id")

                    if tool_call_id:
                        tool_call_names[str(tool_call_id)] = (
                            str(tool_name)
                        )

                    function_call = types.FunctionCall(
                        id=(
                            str(tool_call_id)
                            if tool_call_id is not None
                            else None
                        ),
                        name=str(tool_name),
                        args=dict(arguments),
                    )

                    thought_signature = (
                        raw_tool_call.get("thought_signature")
                        or raw_tool_call.get("thoughtSignature")
                    )

                    parts.append(
                        types.Part(
                            function_call=function_call,
                            thought_signature=thought_signature,
                        )
                    )

                if parts:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=parts,
                        )
                    )

                continue

            # -----------------------------------------------------------------
            # TOOL
            # -----------------------------------------------------------------

            if role == "tool":
                tool_name = (
                    message.get("tool_name")
                    or message.get("name")
                )

                tool_call_id = message.get("tool_call_id")

                if not tool_name and tool_call_id:
                    tool_name = tool_call_names.get(
                        str(tool_call_id)
                    )

                if not tool_name:
                    raise ValueError(
                        "Tool message is missing tool name. "
                        "Provide 'tool_name' or 'name', or make sure "
                        "'tool_call_id' refers to a previous tool call."
                    )

                response_data = cls._normalize_tool_result(
                    message.get("content")
                )

                function_response = types.FunctionResponse(
                    id=(
                        str(tool_call_id)
                        if tool_call_id is not None
                        else None
                    ),
                    name=str(tool_name),
                    response=response_data,
                )

                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part(
                                function_response=function_response
                            )
                        ],
                    )
                )

                continue

        system_instruction = (
            "\n\n".join(system_messages)
            if system_messages
            else None
        )

        return contents, system_instruction

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    def _build_config(
            self,
            system_instruction: str | None,
            tools: Sequence[dict[str, Any]] | None = None,
            stop: Sequence[str] | None = None,
    ) -> types.GenerateContentConfig:
        """
        Build GenerateContentConfig for Gemini.
        """

        gemini_tools = self._build_gemini_tools(tools)

        return types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            stop_sequences=(
                list(stop)
                if stop
                else None
            ),
            system_instruction=system_instruction,
            tools=gemini_tools,

            # We execute tools ourselves.
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
                if gemini_tools
                else None
            ),
        )

    # -------------------------------------------------------------------------
    # RESPONSE PARSING
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_tool_calls(
            response: Any,
    ) -> list[dict[str, Any]]:
        """
        Extract Gemini function calls from the first candidate.

        Thought signatures are preserved.
        """

        candidates = getattr(
            response,
            "candidates",
            None,
        ) or []

        if not candidates:
            return []

        content = getattr(
            candidates[0],
            "content",
            None,
        )

        if content is None:
            return []

        parts = getattr(
            content,
            "parts",
            None,
        ) or []

        result: list[dict[str, Any]] = []

        for index, part in enumerate(parts):
            function_call = getattr(
                part,
                "function_call",
                None,
            )

            if function_call is None:
                continue

            name = getattr(
                function_call,
                "name",
                None,
            )

            if not name:
                continue

            arguments = (
                getattr(
                    function_call,
                    "args",
                    None,
                )
                or {}
            )

            call_id = getattr(
                function_call,
                "id",
                None,
            )

            if not call_id:
                call_id = f"{name}:{index}"

            tool_call: dict[str, Any] = {
                "id": str(call_id),
                "name": str(name),
                "arguments": dict(arguments),
            }

            thought_signature = getattr(
                part,
                "thought_signature",
                None,
            )

            if thought_signature is not None:
                tool_call["thought_signature"] = (
                    thought_signature
                )

            result.append(tool_call)

        return result

    @staticmethod
    def _extract_usage(
            response: Any,
    ) -> tuple[int, int]:
        """
        Extract token usage from Gemini response.
        """

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage is None:
            return 0, 0

        input_tokens = (
            getattr(
                usage,
                "prompt_token_count",
                None,
            )
            or 0
        )

        output_tokens = (
            getattr(
                usage,
                "candidates_token_count",
                None,
            )
            or 0
        )

        return (
            int(input_tokens),
            int(output_tokens),
        )

    # -------------------------------------------------------------------------
    # NORMAL GENERATION
    # -------------------------------------------------------------------------

    async def generate_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> str:
        """
        Generate a normal text response.
        """

        self._validate_messages(messages)

        generic_messages: list[dict[str, Any]] = [
            {
                "role": message.role,
                "content": message.content,
                **(
                    {"tool_calls": message.tool_calls}
                    if message.tool_calls
                    else {}
                ),
                **(
                    {"tool_call_id": message.tool_call_id}
                    if message.tool_call_id
                    else {}
                ),
                **(
                    {"name": message.name}
                    if message.name
                    else {}
                ),
            }
            for message in messages
        ]

        contents, system_instruction = (
            self._messages_to_contents(
                generic_messages
            )
        )

        for message in messages:
            if message.role == "tool":
                raise ValueError(
                    "generate_response() does not support tool messages. "
                    "Use chat() for tool calling."
                )

        config = self._build_config(
            system_instruction=system_instruction,
            tools=None,
            stop=stop,
        )

        async def request() -> Any:
            return await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
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

    # -------------------------------------------------------------------------
    # STREAMING
    # -------------------------------------------------------------------------

    async def get_streaming_response(
            self,
            messages: Sequence[Message],
            stop: Sequence[str] | None = None,
            timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a normal text response.

        Tool calling should go through chat(), not this method.
        """

        self._validate_messages(messages)

        generic_messages: list[dict[str, Any]] = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        for message in messages:
            if message.role == "tool":
                raise ValueError(
                    "get_streaming_response() does not support tool messages. "
                    "Use chat() for tool calling."
                )

        contents, system_instruction = (
            self._messages_to_contents(
                generic_messages
            )
        )

        config = self._build_config(
            system_instruction=system_instruction,
            tools=None,
            stop=stop,
        )

        stream = await (
            self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        )

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

                chunk_text = getattr(
                    chunk,
                    "text",
                    None,
                )

                if not chunk_text:
                    continue

                raw_text += chunk_text

                delta, emitted_text = (
                    self._stream_postprocess_delta(
                        raw_text,
                        emitted_text,
                    )
                )

                if delta:
                    yield delta

        finally:
            close = getattr(
                stream,
                "aclose",
                None,
            )

            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # CHAT / AGENT / TOOLS
    # -------------------------------------------------------------------------

    async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[dict[str, Any]] | None = None,
            timeout: float | None = None,
            stop: Sequence[str] | None = None,
    ) -> Reply:
        """
        Send a chat request to Gemini.

        Function calling is manual:

            1. Send messages + tool declarations.
            2. Gemini returns function calls.
            3. Reply.tool_calls contains those calls.
            4. Agent executes the functions.
            5. Agent sends role="tool" messages in the next request.
            6. Gemini generates the final answer.
        """

        allowed_roles = self.allowed_roles | {"tool"}

        for message in messages:
            role = message.get("role")

            if role not in allowed_roles:
                raise UnknownRoleError(
                    f"{self.__class__.__name__}: Unknown role '{role}'. "
                    f"Allowed roles: {sorted(allowed_roles)}"
                )

        contents, system_instruction = (
            self._messages_to_contents(messages)
        )

        config = self._build_config(
            system_instruction=system_instruction,
            tools=tools,
            stop=stop,
        )

        async def request() -> Any:
            return await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )

        if timeout is not None:
            response = await asyncio.wait_for(
                request(),
                timeout=timeout,
            )
        else:
            response = await request()

        text = self._extract_text(response)

        tool_calls = self._extract_tool_calls(
            response
        )

        input_tokens, output_tokens = (
            self._extract_usage(response)
        )

        return Reply(
            text=self._postprocess(text),
            tool_calls=tool_calls,
            model=self._model_name,
            in_tokens=input_tokens,
            out_tokens=output_tokens,
            cost_usd=0.0,
        )
