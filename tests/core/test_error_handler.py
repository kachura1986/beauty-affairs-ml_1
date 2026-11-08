import logging

import pytest
from fastapi import FastAPI, HTTPException, Path
from fastapi.testclient import TestClient
from pydantic import BaseModel

from core.error_handler import ErrorHandler
from main import app as main_app


class _Payload(BaseModel):
    """Sample model used to trigger an internal Pydantic ValidationError."""

    number: int


@pytest.fixture(scope="module")
def error_app() -> FastAPI:
    """Build an isolated FastAPI app with ErrorHandler and error-raising routes."""
    app = FastAPI()
    ErrorHandler(app, logging.getLogger("tests.error_handler"))

    @app.get("/raise/http/{status_code}")
    async def raise_http(status_code: int) -> None:
        raise HTTPException(status_code=status_code, detail="custom detail")

    @app.get("/raise/value")
    async def raise_value() -> None:
        raise ValueError("bad value")

    @app.get("/raise/unhandled")
    async def raise_unhandled() -> None:
        raise RuntimeError("boom")

    @app.get("/raise/pydantic")
    async def raise_pydantic() -> None:
        # Instantiation itself raises pydantic.ValidationError.
        _Payload(number="not-a-number")

    @app.get("/validated/{value}")
    async def validated(value: int = Path(...)) -> dict:
        return {"value": value}

    return app


@pytest.fixture(scope="module")
def error_client(error_app: FastAPI) -> TestClient:
    # Unhandled exceptions propagate through ServerErrorMiddleware after the
    # 500 response has been sent; disable re-raise so tests can assert on it.
    with TestClient(error_app, raise_server_exceptions=False) as client:
        yield client


def _assert_error_envelope(
        body: dict,
        expected_type: str,
        expected_path: str,
        expected_method: str,
) -> None:
    """Assert that the response body matches the unified error envelope."""
    assert "detail" not in body, "legacy FastAPI error format leaked"
    assert set(body.keys()) == {"error"}
    error = body["error"]
    assert error["type"] == expected_type
    assert error["message"]
    assert error["path"] == expected_path
    assert error["method"] == expected_method


# --- 404 / 405 (P0-2: previously leaked FastAPI's default {"detail": ...}) ---


@pytest.mark.integration
@pytest.mark.regression
def test_404_uses_unified_format_on_main_app() -> None:
    """404 responses must use the unified error envelope (P0-2 regression)."""
    with TestClient(main_app) as client:
        response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    _assert_error_envelope(body, "HTTPException", "/api/does-not-exist", "GET")


@pytest.mark.integration
@pytest.mark.regression
def test_405_uses_unified_format_and_preserves_allow_header_on_main_app() -> None:
    """405 responses must use the unified envelope and preserve the Allow header (P0-2 regression)."""
    with TestClient(main_app) as client:
        response = client.post("/api/health-check")

    assert response.status_code == 405
    body = response.json()
    _assert_error_envelope(body, "HTTPException", "/api/health-check", "POST")
    assert "Allow" in response.headers
    assert "GET" in response.headers["Allow"]


# --- Custom HTTPException ---


@pytest.mark.integration
def test_custom_http_exception_uses_unified_format(error_client: TestClient) -> None:
    response = error_client.get("/raise/http/418")

    assert response.status_code == 418
    body = response.json()
    _assert_error_envelope(body, "HTTPException", "/raise/http/418", "GET")
    assert body["error"]["message"] == "custom detail"


# --- RequestValidationError (client-side 422) ---


@pytest.mark.integration
def test_request_validation_error_uses_unified_format(error_client: TestClient) -> None:
    response = error_client.get("/validated/not-an-int")

    assert response.status_code == 422
    body = response.json()
    _assert_error_envelope(body, "RequestValidationError", "/validated/not-an-int", "GET")
    assert body["error"]["message"] == "Invalid input data in request"
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["details"], "validation details must not be empty"


# --- ValueError (raised by internal code, mapped to 400) ---


@pytest.mark.integration
def test_value_error_uses_unified_format(error_client: TestClient) -> None:
    response = error_client.get("/raise/value")

    assert response.status_code == 400
    body = response.json()
    _assert_error_envelope(body, "ValueError", "/raise/value", "GET")
    assert body["error"]["message"] == "bad value"


# --- Internal pydantic ValidationError (500, no details leaked) ---


@pytest.mark.integration
def test_pydantic_validation_error_uses_unified_format(error_client: TestClient) -> None:
    response = error_client.get("/raise/pydantic")

    assert response.status_code == 500
    body = response.json()
    _assert_error_envelope(body, "InternalServerError", "/raise/pydantic", "GET")
    assert body["error"]["message"] == "Internal server error occurred"
    assert "details" not in body["error"], "internal errors must not leak details"


# --- Unhandled exception (500, no details leaked) ---


@pytest.mark.integration
def test_unhandled_exception_uses_unified_format(error_client: TestClient) -> None:
    response = error_client.get("/raise/unhandled")

    assert response.status_code == 500
    body = response.json()
    _assert_error_envelope(body, "InternalServerError", "/raise/unhandled", "GET")
    assert body["error"]["message"] == "Internal server error occurred"
    assert "details" not in body["error"], "internal errors must not leak details"
