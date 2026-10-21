import uvicorn
from fastapi import FastAPI

from api.router import main_router
from core.config import settings
from core.error_handler import ErrorHandler
from dependencies import get_logger
from lifespan import lifespan


def init_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
    )

    ErrorHandler(app, get_logger())

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
