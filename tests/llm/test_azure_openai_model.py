"""Unit tests for the OpenAI-path transient classifier ``_is_transient_openai_error``."""
import httpx
import openai
import pytest

from services.llm.openai_compatible_model import _is_transient_openai_error

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
    """``httpx.TimeoutException`` cause → permanent (step 1, second half).

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
# APITimeoutError (step 1, first half) — real SDK instance
# ---------------------------------------------------------------------------

def _api_timeout_error() -> openai.APITimeoutError:
    """Build a real ``openai.APITimeoutError`` with no ``__cause__``.

    CONTRACT: ``APITimeoutError.__init__`` requires ``request=`` and does not
    set ``__cause__`` on its own; a plain constructor call therefore
    reproduces the SDK path exactly. ``APITimeoutError`` is an
    ``APIConnectionError`` subclass, so without step 1 it would slip through
    the connection branch (step 3) and be misclassified as transient — this
    is precisely the regression step 1 exists to prevent.
    """
    request = httpx.Request("POST", "https://api.example.test/v1/responses")
    return openai.APITimeoutError(request=request)


def test_real_api_timeout_error_without_cause_is_permanent():
    """Real ``openai.APITimeoutError`` → permanent (step 1, first half).

    CONTRACT: the SDK call was billed; a retry would double-bill. ``__cause__``
    is None (no ``raise ... from ...`` was used), so only step 1 — the
    ``isinstance(exc, APITimeoutError)`` check — can catch it. Without that
    check, step 3 would classify it as transient because ``APITimeoutError``
    is an ``APIConnectionError`` subclass.

    Also pins the deliberate order of step 1 vs step 2: ``APITimeoutError``
    is not a builtin ``TimeoutError`` subclass, so step 2 cannot rescue it.
    Reordering steps 1 and 2 would not change this outcome, but removing
    step 1 entirely would — and no other test fails if step 1 disappears.
    """
    exc = _api_timeout_error()
    assert exc.__cause__ is None
    assert _is_transient_openai_error(exc) is False


def test_api_timeout_error_with_timeout_cause_is_permanent():
    """``openai.APITimeoutError`` with ``httpx.TimeoutException`` cause → permanent.

    CONTRACT: both halves of step 1 agree here. This documents that the two
    checks are not redundant — either one alone would suffice for this shape,
    and the test guards against a future "optimization" that removes one of
    them without understanding the other's coverage.
    """
    exc = _api_timeout_error()
    exc.__cause__ = httpx.ReadTimeout("slow")
    assert _is_transient_openai_error(exc) is False


# ---------------------------------------------------------------------------
# Bare httpx.HTTPStatusError (status code on exc.response.status_code)
# ---------------------------------------------------------------------------

def _httpx_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build a bare httpx.HTTPStatusError carrying the given status code.

    CONTRACT: ``httpx.HTTPStatusError`` does NOT expose ``status_code`` as
    a direct attribute — the code lives on ``exc.response.status_code``. A
    classifier that only reads the direct attribute silently downgrades
    every bare-httpx 429/5xx to permanent. This helper pins that shape so
    the regression fails a test, not a production run.

    ``httpx.HTTPStatusError`` also does NOT subclass ``httpx.TransportError``,
    so the connection-level branch does not cover it — the 429/5xx
    classification must live in step 4.
    """
    request = httpx.Request("POST", "https://api.example.test/v1/responses")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_bare_httpx_status_error_429_and_5xx_is_transient(status_code: int):
    """Bare ``httpx.HTTPStatusError`` 429/5xx is transient — status read from ``exc.response.status_code``."""
    assert _is_transient_openai_error(_httpx_status_error(status_code)) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 408, 422])
def test_bare_httpx_status_error_4xx_is_permanent(status_code: int):
    """Bare ``httpx.HTTPStatusError`` 4xx (except 429) is permanent.

    Pins the boundary of step 4: only 429/5xx are transient; all other
    statuses fall through to permanent — same contract as the SDK path.
    Without this negative control, a future simplification of step 4 to
    ``isinstance(exc, httpx.HTTPStatusError)`` would make 4xx transient
    and no other test would fail.

    ``408 Request Timeout`` is intentionally included: it is semantically
    transient at the HTTP level, but the step 4 contract is "429/5xx only",
    and 408 falls on the permanent side. If someone later widens step 4 to
    ``status in (408, 429) or status >= 500``, this test is the signal that
    the contract is changing.
    """
    assert _is_transient_openai_error(_httpx_status_error(status_code)) is False
