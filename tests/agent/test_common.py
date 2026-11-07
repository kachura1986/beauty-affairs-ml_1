"""Unit tests for the LLM error classifier ``_is_transient_llm_error``."""
import httpx

from services.agent.agent_graph._common import _is_transient_llm_error

from _testing import _FakeAPIStatusError


# ---------------------------------------------------------------------------
# Builtin exceptions
# ---------------------------------------------------------------------------

def test_builtin_timeout_is_transient():
    """Builtin ``TimeoutError`` is classified as transient (retryable)."""
    assert _is_transient_llm_error(TimeoutError("slow")) is True


def test_builtin_connection_is_transient():
    """Builtin ``ConnectionError`` is classified as transient (retryable)."""
    assert _is_transient_llm_error(ConnectionError("refused")) is True


# ---------------------------------------------------------------------------
# status_code-based classification (OpenAI APIStatusError shape)
# ---------------------------------------------------------------------------

def test_429_is_transient():
    """HTTP 429 (rate limit) is transient."""
    assert _is_transient_llm_error(_FakeAPIStatusError(429)) is True


def test_500_is_transient():
    """HTTP 500 is transient."""
    assert _is_transient_llm_error(_FakeAPIStatusError(500)) is True


def test_503_is_transient():
    """HTTP 503 is transient."""
    assert _is_transient_llm_error(_FakeAPIStatusError(503)) is True


def test_400_is_permanent():
    """HTTP 400 (bad request) is permanent — retry would fail identically."""
    assert _is_transient_llm_error(_FakeAPIStatusError(400)) is False


def test_401_is_permanent():
    """HTTP 401 (auth) is permanent."""
    assert _is_transient_llm_error(_FakeAPIStatusError(401)) is False


def test_404_is_permanent():
    """HTTP 404 (not found) is permanent."""
    assert _is_transient_llm_error(_FakeAPIStatusError(404)) is False


# ---------------------------------------------------------------------------
# Arbitrary exceptions
# ---------------------------------------------------------------------------

def test_runtime_error_is_permanent():
    """An unclassified ``RuntimeError`` has no transient signal → permanent (conservative)."""
    assert _is_transient_llm_error(RuntimeError("boom")) is False


# ---------------------------------------------------------------------------
# SDK-wrapped httpx errors
# ---------------------------------------------------------------------------

def test_wrapped_httpx_timeout_is_permanent():
    """``httpx.TimeoutException`` as ``__cause__`` → permanent (APITimeoutError path).

    Mirrors the actual ``openai.APITimeoutError``: the SDK wraps an
    ``httpx.ReadTimeout`` and we must NOT retry it, because a timeout usually
    means the wall-clock budget was consumed by the upstream call.
    """
    exc = Exception("wrapper")
    exc.__cause__ = httpx.ReadTimeout("slow")
    assert _is_transient_llm_error(exc) is False


def test_wrapped_httpx_connect_is_transient():
    """``httpx.ConnectError`` as ``__cause__`` → transient.

    ``ConnectError`` subclasses ``TransportError``, not ``TimeoutException``,
    so it does not hit the timeout guard and reaches ``is_temporary_failure``.
    """
    exc = Exception("wrapper")
    exc.__cause__ = httpx.ConnectError("refused")
    assert _is_transient_llm_error(exc) is True


# ---------------------------------------------------------------------------
# APITimeoutError without __cause__
# ---------------------------------------------------------------------------

def test_ap_timeout_error_without_cause_is_permanent():
    """A timeout-shaped exception without ``__cause__`` is conservative: no retry.

    The classifier only flags ``APITimeoutError`` through
    ``__cause__ = httpx.TimeoutException``. If a future SDK drops the cause,
    nothing else fires (no ``status_code``, no temporary exception name in the
    MRO), so the fallthrough returns False — safe by design.
    """

    class APITimeoutError(Exception):
        pass

    assert _is_transient_llm_error(APITimeoutError()) is False
