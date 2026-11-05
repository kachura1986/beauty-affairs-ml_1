import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.router import main_router
from core.config import settings
from core.error_handler import ErrorHandler
from core.logger import get_logger
from dependencies import PipelineDependencies

# Set up logging handlers before first log record
get_logger()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the application lifecycle.
    Ensures that setup and cleanup occur only within the worker process.
    """
    logger.info(f"Starting {settings.APP_TITLE} server...")

    dependencies = await PipelineDependencies.create()
    app.state.dependencies = dependencies

    try:
        logger.info("Application startup completed.")

        yield

    finally:
        logger.info("Shutting down FastAPI server...")

        await dependencies.close()

        logger.info("Application shutdown completed.")


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

    ErrorHandler(app, logging.getLogger(ErrorHandler.__module__))

    app.include_router(main_router)

    return app


app = init_app()

if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host=settings.HOST,
        port=settings.PORT,
        log_config=None,
    )
