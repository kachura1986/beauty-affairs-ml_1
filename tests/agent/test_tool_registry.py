import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from services.agent.tools import Tool, ToolErrorCode, ToolRegistry, ToolResult, ToolStatus


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


# ---------------------------------------------------------------------------
# ToolResult.args_keys — redaction policy (see ToolRegistry.invoke)
# ---------------------------------------------------------------------------

async def _ok_fn(**kwargs):
    return 'ok'


async def _failing_fn(**kwargs):
    raise RuntimeError('boom')


async def _slow_fn(**kwargs):
    await asyncio.Event().wait()
    return 'unreachable'  # pragma: no cover


def _single_tool_registry(tool_fn, *, name='echo', timeout_seconds=None):
    """Build a minimal registry with exactly one tool."""
    return ToolRegistry(
        [Tool(name=name, description='', fn=tool_fn, args_model=EchoArgs, timeout_seconds=timeout_seconds)],
        default_timeout_seconds=5.0,
        max_output_chars=1000,
    )


async def test_invoke_ok_populates_args_keys():
    """OK path: ``args_keys`` is the sorted set of schema field names."""
    registry = _single_tool_registry(_ok_fn)
    result = await registry.invoke(
        {'name': 'echo', 'arguments': {'value': 'hi'}},
        timeout=5.0,
    )

    assert result.status is ToolStatus.OK
    assert result.args_keys == ('value',)


async def test_invoke_validation_returns_empty_args_keys():
    """VALIDATION path: raw key names are not written (unknown or mistyped)."""
    registry = _single_tool_registry(_ok_fn)
    result = await registry.invoke(
        # `unknown_field` is rejected by ``extra='forbid'``;
        # ``value`` is wrong type. Validation fails before invoke reaches the tool.
        {'name': 'echo', 'arguments': {'value': 123, 'unknown_field': 'x'}},
        timeout=5.0,
    )

    assert result.status is ToolStatus.ERROR
    assert result.error.error_code is ToolErrorCode.VALIDATION
    assert result.args_keys == ()


async def test_invoke_unknown_tool_returns_empty_args_keys():
    """UNKNOWN_TOOL path: no ``args_model`` to intersect with, no keys written."""
    registry = _single_tool_registry(_ok_fn)
    result = await registry.invoke(
        {'name': 'does_not_exist', 'arguments': {'value': 'hi', 'api_token': 'secret'}},
        timeout=5.0,
    )

    assert result.status is ToolStatus.ERROR
    assert result.error.error_code is ToolErrorCode.UNKNOWN_TOOL
    assert result.args_keys == ()


async def test_invoke_execution_failed_keeps_args_keys():
    """EXECUTION_FAILED path: schema validation passed, so key names are safe."""
    registry = _single_tool_registry(_failing_fn)
    result = await registry.invoke(
        {'name': 'echo', 'arguments': {'value': 'hi'}},
        timeout=5.0,
    )

    assert result.status is ToolStatus.ERROR
    assert result.error.error_code is ToolErrorCode.EXECUTION_FAILED
    assert result.args_keys == ('value',)


async def test_invoke_temporary_failure_keeps_args_keys():
    """TEMPORARY_FAILURE path (timeout): schema validation passed, key names are safe."""
    registry = _single_tool_registry(_slow_fn, name='slow', timeout_seconds=0.05)
    result = await registry.invoke(
        {'name': 'slow', 'arguments': {'value': 'hi'}},
    )

    assert result.status is ToolStatus.ERROR
    assert result.error.error_code is ToolErrorCode.TEMPORARY_FAILURE
    assert result.args_keys == ('value',)


def test_trace_tool_writes_args_keys_from_result(trace):
    """Trace.tool records ``res.args_keys`` verbatim (as a JSON list)."""
    trace.tool(
        1,
        {'name': 'echo', 'arguments': {'q': 'hi', 'extra_secret': 'leak'}},
        ToolResult(status=ToolStatus.OK, args_keys=('q',)),
        100,
    )

    tool_records = [r for r in trace.records() if r['t'] == 'tool']
    assert len(tool_records) == 1
    # `extra_secret` never appears: it was excluded by invoke's intersection.
    assert tool_records[0]['args_keys'] == ['q']
    # Raw arguments are still hashed (never written) — invariants preserved.
    assert 'args_hash' in tool_records[0]
    assert 'extra_secret' not in tool_records[0]


def test_trace_tool_default_args_keys_is_empty(trace):
    """Trace.tool with a default-constructed ToolResult writes an empty list."""
    trace.tool(
        1,
        {'name': 'echo', 'arguments': {'q': 'hi'}},
        ToolResult(status=ToolStatus.OK),
        100,
    )

    tool_records = [r for r in trace.records() if r['t'] == 'tool']
    assert tool_records[0]['args_keys'] == []


def test_tool_result_is_hashable():
    """``ToolResult`` remains hashable — ``tuple`` keeps it ``frozen``-safe.

    ``args_keys: tuple[str, ...]`` (not ``list[str]``) is a deliberate choice:
    a list would make ``hash(ToolResult(...))`` raise ``TypeError`` at runtime,
    even though the dataclass is ``frozen=True``.
    """
    assert isinstance(hash(ToolResult(status=ToolStatus.OK)), int)
    assert isinstance(hash(ToolResult(status=ToolStatus.OK, args_keys=('q', 'r'))), int)
    # Same args_keys ⇒ equal hash (tuple equality).
    assert hash(ToolResult(status=ToolStatus.OK, args_keys=('q',))) == hash(
        ToolResult(status=ToolStatus.OK, args_keys=('q',))
    )
