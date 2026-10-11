import logging
import traceback
from collections.abc import Sequence

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError


class ErrorHandler:
    """
    Centralized error handler for FastAPI that provides informative and consistent error responses.
    """

    def __init__(self, app: FastAPI, logger: logging.Logger) -> None:
        """
        Initializes the ErrorHandler and registers error handlers.

        Args:
            app: FastAPI application instance.
            logger: Logger instance used to record error details.
        """
        self._logger = logger
        self._register_error_handlers(app)

    @staticmethod
    def _error_response(
            request: Request,
            error_type: str,
            message: str,
            http_status: int,
            details: dict | Sequence | None = None,
    ) -> JSONResponse:
        """
        Builds a standardized JSON error response.

        Args:
            request: FastAPI request object.
            error_type: Type of the error (e.g. "ValidationError").
            message: Main error message.
            http_status: HTTP status code.
            details: Optional detailed error information.

        Returns:
            JSONResponse: Formatted JSON response containing error details.
        """
        response_content: dict[str, dict[str, object]] = {
            'error': {
                'type': error_type,
                'message': message,
                'path': request.url.path,
                'method': request.method,
            }
        }

        if details is not None:
            response_content['error']['details'] = details

        response = JSONResponse(status_code=http_status, content=response_content)

        return response

    def _register_error_handlers(self, app: FastAPI) -> None:
        """
        Registers all custom error handlers for FastAPI.

        Args:
            app: FastAPI application instance.

        Returns:
            None: This method modifies the app in place.
        """

        @app.exception_handler(HTTPException)
        async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
            """
            Handles FastAPI HTTPException errors.

            Args:
                request: FastAPI request object.
                exc: Raised HTTPException.

            Returns:
                JSONResponse: JSON response describing the HTTP error.
            """
            self._logger.warning(f"HTTP error at {request.url.path}: {exc.detail}")

            response = self._error_response(
                request=request,
                error_type='HTTPException',
                message=str(exc.detail),
                http_status=exc.status_code,
            )

            return response

        @app.exception_handler(RequestValidationError)
        async def handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
            """
            Handles FastAPI request validation errors (e.g., invalid JSON body, query params).
            This occurs when the client sends malformed or incorrect data.

            Args:
                request: FastAPI request object.
                exc: Raised RequestValidationError from FastAPI.

            Returns:
                JSONResponse: JSON response describing the client-side validation error (422).
            """
            self._logger.warning(f"Request validation error for {request.url}: {exc.errors()}")

            response = self._error_response(
                request=request,
                error_type='RequestValidationError',
                message='Invalid input data in request',
                http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                details=exc.errors(),
            )

            return response

        @app.exception_handler(ValidationError)
        async def handle_pydantic_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
            """
            Handles internal Pydantic validation errors.
            This occurs when internal server code tries to instantiate a Pydantic model with invalid data
            (e.g., bad data from a database or external API like AWS Bedrock).

            Args:
                request: FastAPI request object.
                exc: Raised ValidationError from Pydantic.

            Returns:
                JSONResponse: JSON response describing the internal server error (500).
            """
            self._logger.error(f"Internal Pydantic validation error: {exc.errors()}\n{traceback.format_exc()}")

            response = self._error_response(
                request=request,
                error_type='InternalServerError',
                message='Internal server error occurred',
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

            return response

        @app.exception_handler(ValueError)
        async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
            """
            Handles Python ValueError exceptions.

            Args:
                request: FastAPI request object.
                exc: Raised ValueError.

            Returns:
                JSONResponse: JSON response describing the value error.
            """
            self._logger.warning(f"Value error: {exc}")

            response = self._error_response(
                request=request,
                error_type='ValueError',
                message=str(exc),
                http_status=status.HTTP_400_BAD_REQUEST,
            )

            return response

        @app.exception_handler(Exception)
        async def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
            """
            Handles all unexpected exceptions.

            Args:
                request: FastAPI request object.
                exc: Raised Exception.

            Returns:
                JSONResponse: JSON response describing the internal server error.
            """
            traceback_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._logger.error(f"Unhandled exception at {request.url.path}: {exc}\n{traceback_str}")

            response = self._error_response(
                request=request,
                error_type='InternalServerError',
                message='Internal server error occurred',
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

            return response
