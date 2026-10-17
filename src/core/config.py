from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the application.
    """
    model_config = SettingsConfigDict(extra='ignore')

    # Application settings
    APP_TITLE: str = "Demo Backend API"
    DESCRIPTION: str = "API for Demo Project."
    HOST: str = '127.0.0.1'
    PORT: int = 8000

    RUNS_DIR: Path = Path('./runs')

    # LLM settings
    TEMPERATURE: float = 0
    MAX_TOKENS: int = 4096

    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_CHAT_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    OPENAI_API_VERSION: str


settings = Settings()
