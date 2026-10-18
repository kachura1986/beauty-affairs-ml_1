from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.router import main_router
from core.config import settings
from core.error_handler import ErrorHandler
from dependencies import get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the application lifecycle.
    Ensures that setup and cleanup occur only within the worker process.
    """
    logger = get_logger()
    logger.info(f"Starting {settings.APP_TITLE} server...")

    ErrorHandler(app, logger)

    yield

    logger.info("Shutting down FastAPI server...")


def init_app() -> FastAPI:
    """
    Initialize and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
    )

    app.include_router(main_router)

    return app


app = init_app()

if __name__ == '__main__':
    uvicorn.run(
        app='main:app',
        host=settings.HOST,
        port=settings.PORT,
        log_config=None,
    )
