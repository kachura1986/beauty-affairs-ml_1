from typing import Any

from google import genai

from .gemini_model import GeminiModel
from .model import LLMModel


class GeminiProvider:
    """
    Gemini provider based on the modern google-genai SDK.
    """

    def __init__(
            self,
            genai_client: genai.Client,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        self._client = genai_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_model(self, model_name: str) -> LLMModel:
        return GeminiModel(
            model_name=model_name,
            genai_client=self._client,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    async def get_available_models(self) -> list[dict[str, Any]]:
        pager = await self._client.aio.models.list()
        result: list[dict[str, Any]] = []

        async for model in pager:
            supported_actions = (
                    getattr(model, "supported_actions", None) or []
            )

            if (
                    supported_actions
                    and "generateContent" not in supported_actions
            ):
                continue

            resource_name = getattr(model, "name", None) or ""
            model_id = resource_name.rsplit("/", 1)[-1]

            if not model_id:
                continue

            result.append({
                "id": f"gemini:{model_id}",
                "name": (
                    getattr(model, "display_name", None) or model_id
                ),
                "provider": "gemini",
            })

        return result

    async def get_provider_health(self) -> dict[str, Any]:
        try:
            await self._client.aio.models.list(
                config={"page_size": 1}
            )

            return {
                "provider": "gemini",
                "status": "healthy",
            }

        except Exception as exc:
            return {
                "provider": "gemini",
                "status": "unhealthy",
                "error": str(exc),
            }
