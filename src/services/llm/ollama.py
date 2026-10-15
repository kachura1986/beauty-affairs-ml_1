from typing import Any

from ollama import AsyncClient

from .model import LLMModel
from .ollama_model import OllamaModel


class OllamaProvider:
    """
    Implement the provider management for the Ollama API using an asynchronous client."""

    def __init__(
            self,
            ollama_client: AsyncClient,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        self._client = ollama_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_model(self, model_name: str) -> LLMModel:
        return OllamaModel(
            model_name=model_name,
            ollama_client=self._client,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    async def get_available_models(self) -> list[dict[str, Any]]:
        response = await self._client.list()
        result: list[dict[str, Any]] = []

        for model in response.models:
            model_id = (
                    getattr(model, "model", None)
                    or getattr(model, "name", None)
            )

            if not model_id:
                continue

            result.append({
                "id": f"ollama:{model_id}",
                "name": model_id,
                "provider": "ollama",
            })

        return result

    async def get_provider_health(self) -> dict[str, Any]:
        try:
            await self._client.list()

            return {
                "provider": "ollama",
                "status": "healthy",
            }

        except Exception as exc:
            return {
                "provider": "ollama",
                "status": "unhealthy",
                "error": str(exc),
            }
