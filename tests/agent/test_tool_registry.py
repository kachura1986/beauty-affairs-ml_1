import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from services.agent.tools import Tool, ToolErrorCode, ToolRegistry, ToolStatus


# ---------------------------------------------------------------------------
# Fake infrastructure
# ---------------------------------------------------------------------------

class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "ok"


# ---------------------------------------------------------------------------
# ToolRegistry.invoke: per-tool timeouts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_timeout, caller_timeout", [
    (0.05, None),
    (0.05, 10.0),
    (10.0, 0.05),
], ids=["per_tool_only", "per_tool_dominates", "caller_dominates"])
async def test_timeout_dominance(tool_timeout, caller_timeout):
    """The effective timeout is the minimum of the applicable ceilings.

    Three scenarios, sharing the same assert set because the observable
    outcome is identical — the tool must not run longer than the tightest
    applicable timeout and must surface as a temporary failure:

      - per_tool_only:    only the per-tool cap applies (caller passes None).
      - per_tool_dominates: per-tool cap (0.05s) < caller (10s); min wins.
      - caller_dominates:  caller (0.05s) < per-tool cap (10s); min wins.
    """

    async def hanging_fn(**kwargs):
        await asyncio.sleep(10)

    registry = ToolRegistry(
        [
            Tool(
                name="hang",
                description="",
                fn=hanging_fn,
                args_model=EchoArgs,
                timeout_seconds=tool_timeout,
            ),
        ],
        default_timeout_seconds=30.0,
        max_output_chars=15000,
    )

    res = await registry.invoke({"name": "hang", "arguments": {}}, timeout=caller_timeout)

    assert res.status is ToolStatus.ERROR
    assert res.error.error_code is ToolErrorCode.TEMPORARY_FAILURE
    assert res.retryable is True


async def test_quick_call_completes_within_per_tool_cap():
    """A tool that finishes before its per-tool cap returns a normal result."""

    async def quick_fn(**kwargs):
        await asyncio.sleep(0.01)
        return "ok"

    registry = ToolRegistry(
        [
            Tool(
                name="quick",
                description="",
                fn=quick_fn,
                args_model=EchoArgs,
                timeout_seconds=0.5,
            ),
        ],
        default_timeout_seconds=30.0,
        max_output_chars=15000,
    )

    res = await registry.invoke({"name": "quick", "arguments": {}})

    assert res.status is ToolStatus.OK
    assert res.content
    assert res.error is None


async def test_cancelled_error_propagates():
    """CancelledError from the tool is not swallowed by the registry.

    ``asyncio.CancelledError`` is a BaseException, not an Exception, so
    ``except Exception`` in invoke() must not catch it. External cancellation
    (FastAPI shutdown, parent coroutine timeout) must propagate.
    """

    async def cancelled_fn(**kwargs):
        raise asyncio.CancelledError()

    registry = ToolRegistry(
        [
            Tool(
                name="cancelled",
                description="",
                fn=cancelled_fn,
                args_model=EchoArgs,
                timeout_seconds=1.0,
            ),
        ],
        default_timeout_seconds=30.0,
        max_output_chars=15000,
    )

    with pytest.raises(asyncio.CancelledError):
        await registry.invoke({"name": "cancelled", "arguments": {}})


# ---------------------------------------------------------------------------
# ToolRegistry.__init__: rejects sync callables at registration
# ---------------------------------------------------------------------------

def test_registry_rejects_sync_function_at_registration():
    """A sync callable must fail loud at ToolRegistry construction.

    Regression guard: the registry only supports async tools. If the
    fail-loud check in __init__ is removed, a sync tool would register
    silently and hang/misbehave at invoke() time instead of failing
    at startup.
    """

    def sync_fn(value: str = "ok"):
        return value

    with pytest.raises(TypeError, match=r"must be an async callable") as exc_info:
        ToolRegistry(
            [Tool(name="sync", description="", fn=sync_fn, args_model=EchoArgs)],
            default_timeout_seconds=30.0,
            max_output_chars=15000,
        )
    assert "sync" in str(exc_info.value)
    assert "async" in str(exc_info.value).lower()


def test_registry_rejects_sync_function_even_alongside_async():
    """A single sync tool must abort registration even if async tools are valid.

    Pins that the fail-loud check runs per-tool during __init__, not
    lazily at invoke() time, and that a valid async tool earlier in the
    iterable does not mask the problem.
    """

    def sync_fn(value: str = "ok"):
        return value

    async def async_fn(value: str = "ok"):
        return value

    with pytest.raises(TypeError, match=r"must be an async callable"):
        ToolRegistry(
            [
                Tool(name="good", description="", fn=async_fn, args_model=EchoArgs),
                Tool(name="bad", description="", fn=sync_fn, args_model=EchoArgs),
            ],
            default_timeout_seconds=30.0,
            max_output_chars=15000,
        )
