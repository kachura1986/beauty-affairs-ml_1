"""Unit tests for the OpenAI-path transient classifier ``_is_transient_openai_error``."""
import httpx

from services.llm.azure_openai_model import _is_transient_openai_error

from _testing import _FakeAPIStatusError


# ---------------------------------------------------------------------------
# Builtin exceptions
# ---------------------------------------------------------------------------

def test_builtin_timeout_is_transient():
    """Builtin TimeoutError from asyncio.wait_for is the production timeout path."""
    assert _is_transient_openai_error(TimeoutError("slow")) is True


def test_builtin_connection_is_transient():
    """Builtin ConnectionError is transient (retryable)."""
    assert _is_transient_openai_error(ConnectionError("refused")) is True


# ---------------------------------------------------------------------------
# status_code-based classification (OpenAI APIStatusError shape)
# ---------------------------------------------------------------------------

def test_429_is_transient():
    """HTTP 429 (rate limit) is transient."""
    assert _is_transient_openai_error(_FakeAPIStatusError(429)) is True


def test_500_is_transient():
    """HTTP 500 is transient."""
    assert _is_transient_openai_error(_FakeAPIStatusError(500)) is True


def test_503_is_transient():
    """HTTP 503 is transient."""
    assert _is_transient_openai_error(_FakeAPIStatusError(503)) is True


def test_400_is_permanent():
    """HTTP 400 (bad request) is permanent — retry would fail identically."""
    assert _is_transient_openai_error(_FakeAPIStatusError(400)) is False


def test_401_is_permanent():
    """HTTP 401 (auth) is permanent."""
    assert _is_transient_openai_error(_FakeAPIStatusError(401)) is False


def test_404_is_permanent():
    """HTTP 404 (not found) is permanent."""
    assert _is_transient_openai_error(_FakeAPIStatusError(404)) is False


# ---------------------------------------------------------------------------
# Arbitrary exceptions
# ---------------------------------------------------------------------------

def test_runtime_error_is_permanent():
    """Unclassified exceptions fall through to permanent (conservative)."""
    assert _is_transient_openai_error(RuntimeError("boom")) is False


# ---------------------------------------------------------------------------
# SDK-wrapped httpx errors
# ---------------------------------------------------------------------------

def test_wrapped_httpx_timeout_is_permanent():
    """``httpx.TimeoutException`` cause → permanent.

    CONTRACT: the SDK call was billed; a retry would double-bill.
    """
    exc = Exception("wrapper")
    exc.__cause__ = httpx.ReadTimeout("slow")
    assert _is_transient_openai_error(exc) is False


def test_wrapped_httpx_connect_is_transient():
    """``httpx.ConnectError`` cause → transient.

    CONTRACT: ``ConnectError`` subclasses ``TransportError``, not
    ``TimeoutException``, so it reaches the connection-level branch.
    """
    exc = Exception("wrapper")
    exc.__cause__ = httpx.ConnectError("refused")
    assert _is_transient_openai_error(exc) is True


# ---------------------------------------------------------------------------
# APITimeoutError without __cause__
# ---------------------------------------------------------------------------

def test_ap_timeout_error_without_cause_is_permanent():
    """Timeout-shaped exception without ``__cause__`` falls through to permanent.

    CONTRACT: no ``status_code`` and no temporary exception name in the MRO →
    fallthrough returns False. A dropped cause must never trigger a retry of
    an already-consumed budget.
    """

    class APITimeoutError(Exception):
        pass

    assert _is_transient_openai_error(APITimeoutError()) is False
