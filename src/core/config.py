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

    # Azure
    # OpenAI
    AZURE_OPENAI_API_KEY: SecretStr
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    # Searching
    SEARCH_API_KEY: SecretStr
    SEARCH_ENDPOINT: str

    # Agent configuration
    AGENTS_CONFIG: Path = Path(__file__).resolve().parents[1] / 'agents.yaml'
    # Maximum number of concurrent agent runs per worker
    MAX_CONCURRENT_AGENT_RUNS: int = Field(default=4, gt=0)

    # LLM
    # Per-operation caps; the run budget remains the outer limit
    LLM_DEFAULT_TIMEOUT_SECONDS: float = 120.0
    # Pre-flight threshold: do not start an LLM call when remaining tokens are strictly below this value
    MIN_EXPECTED_TOKENS: int = 500
    TOOL_DEFAULT_TIMEOUT_SECONDS: float = 30.0
    TOOL_DEFAULT_MAX_OUTPUT_CHARS: int = 15000
    # Retry
    LLM_RETRY_BACKOFF_SECONDS: float = 1.0
    LLM_RETRY_MIN_TIMEOUT_SECONDS: float = 5.0

    # Embedding
    EMBEDDING_DIMENSIONS: int = 1024


settings = Settings()
