from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import settings
from dependencies import PipelineDependencies, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the application lifecycle.
    Ensures that setup and cleanup occur only within the worker process.
    """
    logger = get_logger()
    logger.info("Starting %s server...", settings.APP_TITLE)

    dependencies = await PipelineDependencies.create()
    app.state.dependencies = dependencies

    try:
        logger.info("Application startup completed.")

        yield

    finally:
        logger.info("Shutting down FastAPI server...")

        await dependencies.close()

        logger.info("Application shutdown completed.")
