from typing import Any, Protocol

from .model import LLMModel


class LLMProvider(Protocol):
    """
    Define the application-level interface for asynchronous LLM providers.
    """

    async def get_available_models(self) -> list[dict[str, Any]]:
        ...

    async def get_provider_health(self) -> dict[str, Any]:
        ...

    def get_model(self, model_name: str) -> LLMModel:
        ...
