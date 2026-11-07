import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict

from services.agent.agent_graph.state import AgentContext
from services.agent.context import AgentContextWindow
from services.agent.tools import Tool
from services.llm import Reply


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "ok"


async def echo_fn(value: str = "ok"):
    return value


async def failing_fn(**kwargs):
    raise RuntimeError("internal boom")


async def timeout_fn(**kwargs):
    raise TimeoutError("temporary timeout")


class _FakeAPIStatusError(Exception):
    """Stands in for openai.APIStatusError subclasses without depending on the SDK."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"status_{status_code}")


def rate_limit_error() -> Exception:
    """Build a 429 exception as raised by the OpenAI SDK."""
    return _FakeAPIStatusError(429)


def internal_server_error(status_code: int = 503) -> Exception:
    """Build a 5xx exception as raised by the OpenAI SDK."""
    return _FakeAPIStatusError(status_code)


class FakeLLM:
    """Protocol-compatible LLM: returns canned replies or raises scripted errors."""

    def __init__(
            self,
            *replies: Reply,
            errors: list[Exception] | None = None,
            raise_after: float | None = None,
    ):
        self._replies = list(replies)
        self._errors = list(errors or [])
        self._raise_after = raise_after
        self.calls: list[dict] = []
        self.model_name = "fake"

    async def chat(self, messages, tools=None, timeout=None, system_prompt=None):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "timeout": timeout,
            "system_prompt": system_prompt,
        })
        if self._errors:
            # Optional delay before raising, so latency-sensitive tests can observe
            # a non-zero duration in the error path. Never applied on the success path.
            if self._raise_after is not None:
                await asyncio.sleep(self._raise_after)
            raise self._errors.pop(0)
        return self._replies.pop(0)


class FakeRuntime:
    """Minimal Runtime[AgentContext] stand-in."""

    def __init__(self, context: AgentContext):
        self.context = context


def reply(text="ok", tool_calls=None, in_tokens=0, out_tokens=0, cost_usd=0.0, model="fake"):
    """Helper to reduce noise and make tests resilient to Reply changes."""
    return Reply(
        text=text,
        tool_calls=tool_calls or [],
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost_usd=cost_usd,
        model=model,
    )


def failing_tool() -> Tool:
    return Tool(name="failing", description="", fn=failing_fn, args_model=EchoArgs)


def timeout_tool() -> Tool:
    return Tool(name="timeout", description="", fn=timeout_fn, args_model=EchoArgs)


def make_context_window(
        system: str = "sys",
        messages: list[dict[str, Any]] | None = None,
) -> AgentContextWindow:
    return AgentContextWindow(system=system, messages=messages or [])
