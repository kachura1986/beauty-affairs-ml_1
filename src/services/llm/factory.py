from google import genai
from ollama import AsyncClient
from openai import AsyncAzureOpenAI

from .azure_openai import AzureOpenAIProvider
from .gemini import GeminiProvider
from .model import LLMModel
from .ollama import OllamaProvider
from .provider import LLMProvider


def create_llm_provider(
        provider_name: str,
        ollama_client: AsyncClient,
        genai_client: genai.Client | None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        azure_openai_client: AsyncAzureOpenAI | None = None,
        azure_openai_chat_model: str | None = None,
) -> LLMProvider:
    """
    Instantiates a specific LLMProvider implementation based on provider_name.

    Args:
        provider_name (str): Provider identifier (e.g. "gemini", "ollama" or "azure_openai").
        ollama_client (AsyncClient): Ollama client instance.
        genai_client (genai.Client | None): Gemini client instance.
        temperature (float, optional): Temperature for the LLM. Defaults to 0.0.
        max_tokens (int, optional): Maximum tokens for the LLM. Defaults to 4096.
        azure_openai_client (AsyncAzureOpenAI | None): Azure OpenAI client instance.
        azure_openai_chat_model (str | None): Azure OpenAI deployment name.

    Returns:
        LLMProvider: An instance of the appropriate LLMProvider implementation.

    Raises:
        ValueError: If the provider is not supported or is not configured.
    """

    if provider_name == 'gemini':
        if genai_client is None:
            raise ValueError("Gemini provider is not configured")

        return GeminiProvider(
            genai_client=genai_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider_name == 'ollama':
        return OllamaProvider(
            ollama_client=ollama_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider_name == 'azure_openai':
        if azure_openai_client is None or not azure_openai_chat_model:
            raise ValueError("Azure OpenAI provider is not configured")

        return AzureOpenAIProvider(
            azure_openai_client=azure_openai_client,
            chat_model=azure_openai_chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unsupported provider: '{provider_name}'")


def create_llm_model(
        model_id: str,
        ollama_client: AsyncClient,
        genai_client: genai.Client | None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        azure_openai_client: AsyncAzureOpenAI | None = None,
        azure_openai_chat_model: str | None = None,
) -> LLMModel:
    """
    Instantiates a specific LLMModel from an identifier in format "provider:model_name".

    Args:
        model_id (str): Identifier in format "provider:model_name".
        ollama_client (AsyncClient): Ollama client instance.
        genai_client (genai.Client | None): Gemini client instance.
        temperature (float, optional): Temperature for the LLM. Defaults to 0.0.
        max_tokens (int, optional): Maximum tokens for the LLM. Defaults to 4096.
        azure_openai_client (AsyncAzureOpenAI | None): Azure OpenAI client instance.
        azure_openai_chat_model (str | None): Azure OpenAI deployment name.

    Returns:
        LLMModel: An instance of the requested model.

    Raises:
        ValueError: If the model_id format is invalid or if the provider is not supported.
    """

    if ":" not in model_id:
        raise ValueError(f"Invalid model_id format: '{model_id}'. Expected format 'provider:model_name'.")

    provider_name, model_name = model_id.split(":", 1)

    provider = create_llm_provider(
        provider_name=provider_name,
        ollama_client=ollama_client,
        genai_client=genai_client,
        temperature=temperature,
        max_tokens=max_tokens,
        azure_openai_client=azure_openai_client,
        azure_openai_chat_model=azure_openai_chat_model,
    )

    return provider.get_model(model_name)
