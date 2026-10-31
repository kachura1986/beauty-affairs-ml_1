from pathlib import Path

from pydantic import Field, SecretStr
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
    MAX_TOKENS: int = Field(default=8192, gt=0)
    # A value of 0 explicitly means the model is free: the run proceeds as long as no nonzero cost is recorded,
    # and terminates as soon as any nonzero cost is reported (see Budget.exceeded_without_steps).
    INPUT_COST_PER_MILLION: float = Field(ge=0)
    OUTPUT_COST_PER_MILLION: float = Field(ge=0)

    AZURE_OPENAI_API_KEY: SecretStr
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_CHAT_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str

    # Searching
    SEARCH_API_KEY: SecretStr
    SEARCH_ENDPOINT: str

    # Agent configuration
    AGENTS_CONFIG: Path = Path(__file__).resolve().parents[1] / 'config' / 'agents.yaml'

    # Timeouts (per-operation caps; the run budget remains the outer limit)
    LLM_DEFAULT_TIMEOUT_SECONDS: float = 120.0
    TOOL_DEFAULT_TIMEOUT_SECONDS: float = 30.0

    # Pre-flight threshold: do not start an LLM call when remaining tokens
    # are strictly below this value. Boundary (==) is allowed.
    MIN_EXPECTED_TOKENS: int = 500


settings = Settings()
