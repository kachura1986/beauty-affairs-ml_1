from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the application.
    """
    model_config = SettingsConfigDict(extra='ignore')

    # Application settings
    APP_TITLE: str = "Beauty Affair ML API"
    DESCRIPTION: str = "API for Beauty Affair ML."
    HOST: str = '127.0.0.1'
    PORT: int = 8000



settings = Settings()
