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

async def test_per_tool_cap_triggers_temporary_failure():
    """A tool exceeding its per-tool cap yields TEMPORARY_FAILURE, retryable=True.

    ``asyncio.wait_for`` raises TimeoutError on the per-tool ceiling; the
    registry classifies it as a temporary failure via errors.py.
    """

    async def hanging_fn(**kwargs):
        await asyncio.sleep(10)

    registry = ToolRegistry([
        Tool(
            name="hang",
            description="",
            fn=hanging_fn,
            args_model=EchoArgs,
            timeout_seconds=0.05,
        ),
    ])

    res = await registry.invoke({"name": "hang", "arguments": {}})

    assert res.status is ToolStatus.ERROR
    assert res.error.error_code is ToolErrorCode.TEMPORARY_FAILURE
    assert res.retryable is True


async def test_per_tool_cap_dominates_caller_timeout():
    """The per-tool ceiling wins when it is smaller than the caller timeout.

    A tool declared with timeout_seconds=0.05 must not run for 10s just
    because the caller allowed it: min(0.05, 10) == 0.05.
    """

    async def hanging_fn(**kwargs):
        await asyncio.sleep(10)

    registry = ToolRegistry([
        Tool(
            name="hang",
            description="",
            fn=hanging_fn,
            args_model=EchoArgs,
            timeout_seconds=0.05,
        ),
    ])

    res = await registry.invoke({"name": "hang", "arguments": {}}, timeout=10.0)

    assert res.error.error_code is ToolErrorCode.TEMPORARY_FAILURE


async def test_caller_timeout_dominates_per_tool_cap():
    """The caller timeout wins when it is smaller than the per-tool ceiling.

    A tool declared with timeout_seconds=10 must still be capped at 0.05s
    when the caller only has 0.05s of budget left: min(10, 0.05) == 0.05.
    """

    async def hanging_fn(**kwargs):
        await asyncio.sleep(10)

    registry = ToolRegistry([
        Tool(
            name="hang",
            description="",
            fn=hanging_fn,
            args_model=EchoArgs,
            timeout_seconds=10.0,
        ),
    ])

    res = await registry.invoke({"name": "hang", "arguments": {}}, timeout=0.05)

    assert res.error.error_code is ToolErrorCode.TEMPORARY_FAILURE


async def test_quick_call_completes_within_per_tool_cap():
    """A tool that finishes before its per-tool cap returns a normal result."""

    async def quick_fn(**kwargs):
        await asyncio.sleep(0.01)
        return "ok"

    registry = ToolRegistry([
        Tool(
            name="quick",
            description="",
            fn=quick_fn,
            args_model=EchoArgs,
            timeout_seconds=0.5,
        ),
    ])

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

    registry = ToolRegistry([
        Tool(
            name="cancelled",
            description="",
            fn=cancelled_fn,
            args_model=EchoArgs,
            timeout_seconds=1.0,
        ),
    ])

    with pytest.raises(asyncio.CancelledError):
        await registry.invoke({"name": "cancelled", "arguments": {}})
