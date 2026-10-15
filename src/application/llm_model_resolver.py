import logging
import time
from asyncio import gather
from typing import Any

from google import genai
from ollama import AsyncClient

from services.llm import LLMModel, create_llm_model, create_llm_provider


class LLMModelResolver:
    def __init__(
            self,
            ollama_client: AsyncClient,
            genai_client: genai.Client | None,
            ollama_host: str,
            default_temperature: float = 0.0,
            default_max_tokens: int = 4096,
    ) -> None:
        """Initialize the model resolver with required clients and parameters."""
        self._ollama_client = ollama_client
        self._genai_client = genai_client
        self._ollama_host = ollama_host
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._logger = logging.getLogger(__name__)

    def resolve(self, model_id: str) -> LLMModel:
        return create_llm_model(
            model_id=model_id,
            ollama_client=self._ollama_client,
            genai_client=self._genai_client,
            temperature=self._default_temperature,
            max_tokens=self._default_max_tokens,
        )

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Dynamically fetches available models from both Ollama and Gemini APIs."""

        async def fetch_gemini_models() -> list[dict[str, Any]]:
            if self._genai_client is None:
                return []

            try:
                provider = create_llm_provider(
                    provider_name="gemini",
                    ollama_client=self._ollama_client,
                    genai_client=self._genai_client,
                    temperature=self._default_temperature,
                    max_tokens=self._default_max_tokens,
                )
                models = await provider.get_available_models()

                return [
                    {**model, "type": "cloud"}
                    for model in models
                ]

            except Exception:
                self._logger.exception("Failed to fetch Gemini models")
                return []

        async def fetch_ollama_models() -> list[dict[str, Any]]:
            try:
                provider = create_llm_provider(
                    provider_name="ollama",
                    ollama_client=self._ollama_client,
                    genai_client=self._genai_client,
                    temperature=self._default_temperature,
                    max_tokens=self._default_max_tokens,
                )
                models = await provider.get_available_models()

                return [
                    {**model, "name": f"{model['name']} (Local)", "type": "local"}
                    for model in models
                ]

            except Exception:
                self._logger.exception("Failed to fetch Ollama models")
                return []

        gemini_models, ollama_models = await gather(
            fetch_gemini_models(),
            fetch_ollama_models(),
        )

        return [*gemini_models, *ollama_models]

    async def get_provider_health(self) -> dict[str, Any]:
        """Checks connectivity of configured LLM providers."""

        async def check_provider(provider_name: str) -> dict[str, Any]:
            if provider_name == "gemini" and self._genai_client is None:
                return {
                    "provider": "gemini",
                    "status": "not_configured",
                    "latency_ms": 0,
                }

            started = time.perf_counter()

            try:
                provider = create_llm_provider(
                    provider_name=provider_name,
                    ollama_client=self._ollama_client,
                    genai_client=self._genai_client,
                    temperature=self._default_temperature,
                    max_tokens=self._default_max_tokens,
                )
                health = await provider.get_provider_health()
                health.setdefault(
                    "latency_ms",
                    round((time.perf_counter() - started) * 1000),
                )
                return health

            except Exception as exc:
                self._logger.warning(
                    "%s health check failed: %s",
                    provider_name,
                    exc,
                )
                return {
                    "provider": provider_name,
                    "status": "unhealthy",
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                }

        ollama_health, gemini_health = await gather(
            check_provider("ollama"),
            check_provider("gemini"),
        )

        return {
            "status": (
                "healthy"
                if all(
                    health["status"] in {"healthy", "not_configured"}
                    for health in (ollama_health, gemini_health)
                )
                else "degraded"
            ),
            "providers": {
                "ollama": ollama_health,
                "gemini": gemini_health,
            },
        }
