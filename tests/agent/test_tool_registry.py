import asyncio
import json

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
    """The tool must not run longer than the tightest applicable timeout and must surface as a temporary failure."""

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
    """External cancellation (shutdown, parent timeout) must propagate, not be swallowed as a tool error."""

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
    """A sync tool must fail at construction — otherwise it would silently bypass the per-tool timeout at invoke()."""

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
    """The fail-loud check runs per-tool at construction; a valid async tool must not mask a sync one."""

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


async def test_trace_tool_does_not_leak_raw_field_names_via_hint(trace):
    """Regression: extra_forbidden puts the raw field name into the model-facing hint; it must not reach the trace."""
    registry = _single_tool_registry(_ok_fn)
    call = {'name': 'echo', 'arguments': {'value': 'hi', 'api_token': 'secret-value'}}
    result = await registry.invoke(call)

    assert result.status is ToolStatus.ERROR
    assert result.error.error_code is ToolErrorCode.VALIDATION
    # Model must see the field name to be able to correct the call.
    assert 'api_token' in result.error.hint

    trace.tool(1, call, result, 100)
    record = [r for r in trace.records() if r['t'] == 'tool'][0]
    serialized = json.dumps(record)
    assert 'api_token' not in serialized
    assert 'secret-value' not in serialized


def test_trace_tool_never_records_raw_argument_values(trace):
    """Raw argument VALUES must not appear literally in the trace — only args_hash and args_keys."""
    call = {'name': 'echo', 'arguments': {'value': 'RAW-VALUE-MARKER-12345'}}
    trace.tool(1, call, ToolResult(status=ToolStatus.OK, args_keys=('value',)), 100)

    record = [r for r in trace.records() if r['t'] == 'tool'][0]
    serialized = json.dumps(record)
    assert 'RAW-VALUE-MARKER-12345' not in serialized
    assert 'args_hash' in record
    assert record['args_keys'] == ['value']  # key names — yes; values — no


def test_trace_tool_args_hash_is_deterministic_and_low_entropy_recoverable(trace):
    """
    args_hash is a deterministic SHA-256, not a cryptographic commitment: low-entropy values (top_k, min_price)
    are recoverable by enumeration. High-entropy values are not. This is a documented trade-off — determinism
    across runs is required for post-mortem diffing.
    """
    import hashlib
    import json

    def _hash(obj):
        # Mirrors Trace.tool(): hashes only `arguments`, NOT {'n': ..., 'a': ...}.
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # Determinism: same args → same hash across calls.
    call_a = {'name': 'search_products', 'arguments': {'query': 'vitamin c', 'top_k': 10}}
    call_b = {'name': 'search_products', 'arguments': {'query': 'vitamin c', 'top_k': 10}}
    trace.tool(1, call_a, ToolResult(status=ToolStatus.OK), 10)
    trace.tool(2, call_b, ToolResult(status=ToolStatus.OK), 10)
    records = [r for r in trace.records() if r['t'] == 'tool']
    assert records[0]['args_hash'] == records[1]['args_hash']

    # Low-entropy recoverable: top_k=10 and top_k=20 produce different hashes,
    # and each matches the recomputed SHA-256 — enumeration works.
    call_c = {'name': 'search_products', 'arguments': {'query': 'vitamin c', 'top_k': 20}}
    trace.tool(3, call_c, ToolResult(status=ToolStatus.OK), 10)
    records = [r for r in trace.records() if r['t'] == 'tool']
    assert records[0]['args_hash'] != records[2]['args_hash']
    assert records[2]['args_hash'] == _hash(call_c['arguments'])

    # Raw values themselves are still absent.
    serialized = json.dumps(records[0])
    assert '"top_k": 10' not in serialized


def test_tool_result_is_hashable():
    """A ``list`` would break ``hash(ToolResult(...))`` despite ``frozen=True``; ``tuple`` keeps it hashable."""
    assert isinstance(hash(ToolResult(status=ToolStatus.OK)), int)
    assert isinstance(hash(ToolResult(status=ToolStatus.OK, args_keys=('q', 'r'))), int)
    # Same args_keys ⇒ equal hash (tuple equality).
    assert hash(ToolResult(status=ToolStatus.OK, args_keys=('q',))) == hash(
        ToolResult(status=ToolStatus.OK, args_keys=('q',))
    )
