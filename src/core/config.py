from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the application."""
    model_config = SettingsConfigDict(extra='ignore')

    # Application settings
    APP_TITLE: str = "Beauty Affairs Agent API"
    DESCRIPTION: str = (
        "Provides personalized skincare recommendations based on user skin type, concerns, "
        "and skincare goals."
    )
    HOST: str = '127.0.0.1'
    PORT: int = 8000

    # Temporary files
    TEMP_DIR: Path = Path(__file__).resolve().parents[2] / '.temp'
    RUNS_DIR: Path = TEMP_DIR / 'runs'
    LOGS_DIR: Path = TEMP_DIR / 'logs'

    # Embedding
    EMBEDDING_DIMENSIONS: int = 1024

    # LLM settings
    TEMPERATURE: float = 0
    MAX_TOKENS: int = 4096
    INPUT_COST_PER_MILLION: float = 0
    OUTPUT_COST_PER_MILLION: float = 0

    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_CHAT_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str

    # Searching
    SEARCH_API_KEY: str
    SEARCH_ENDPOINT: str


settings = Settings()
