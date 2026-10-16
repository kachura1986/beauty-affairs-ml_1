from typing import Any

from openai import AsyncAzureOpenAI

from .azure_openai_model import AzureOpenAIModel
from .model import LLMModel


class AzureOpenAIProvider:
    """
    Implement the provider management for the Azure OpenAI API using an asynchronous client.
    """

    def __init__(
            self,
            azure_openai_client: AsyncAzureOpenAI,
            chat_model: str,
            temperature: float = 0.0,
            max_tokens: int = 4096,
    ) -> None:
        self._client = azure_openai_client
        self._chat_model = chat_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_model(self, model_name: str) -> LLMModel:
        return AzureOpenAIModel(
            model_name=model_name,
            azure_openai_client=self._client,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    async def get_available_models(self) -> list[dict[str, Any]]:
        if not self._chat_model:
            return []

        return [
            {
                'id': f'azure_openai:{self._chat_model}',
                'name': self._chat_model,
                'provider': 'azure_openai',
            }
        ]

    async def get_provider_health(self) -> dict[str, Any]:
        try:
            await self._client.models.list()

            return {
                'provider': 'azure_openai',
                'status': 'healthy',
            }

        except Exception as exc:
            return {
                'provider': 'azure_openai',
                'status': 'unhealthy',
                'error': str(exc),
            }
