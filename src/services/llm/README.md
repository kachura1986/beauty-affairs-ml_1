# LLM Package Guide

## Module Overview

| Module | Responsibility |
|---|---|
| `model.py` | `LLMModel` protocol defining the model interface. |
| `provider.py` | `LLMProvider` protocol defining the provider interface. |
| `base_model.py` | `BaseLLMModel` with shared validation and response post-processing. |
| `schemas.py` | Shared data structures: `Message`, `Reply`, roles, and exceptions. |
| `factory.py` | Creates providers and models from provider names or `provider:model_name` IDs. |
| `gemini.py` | `GeminiProvider`: model creation, model listing, and health checks. |
| `gemini_model.py` | `GeminiModel`: Gemini API calls, streaming, message conversion, and tool calling. |
| `ollama.py` | `OllamaProvider`: model creation, model listing, and health checks. |
| `ollama_model.py` | `OllamaModel`: Ollama API calls, streaming, and tool calling. |
| `__init__.py` | Public package exports. |

## Adding a New Provider

For example, to add an `OpenAI` provider:

1. Create `openai.py` and `openai_model.py`.
2. Implement `OpenAIModel` by inheriting from `BaseLLMModel`.
3. Implement `OpenAIProvider` with the following methods:
   - `get_model(model_name)`
   - `get_available_models()`
   - `get_provider_health()`
4. Register the provider in `factory.py`.
5. Export the new classes from `__init__.py`.

### Implement the Model

Inherit from `BaseLLMModel` and implement all required methods:

- `model_name`
- `generate_response()`
- `get_streaming_response()`
- `chat()`

Use `Message` and `Reply` for consistent application-level data structures. Convert messages, tool schemas, and API responses to and from the provider's native format. Reuse `_validate_messages()`, `_postprocess()`, and `_stream_postprocess_delta()` where appropriate.

### Implement the Provider

The provider should store its API client and shared configuration, create model instances, list available models, and report provider health.

### Register the Provider

Add the new provider to `create_llm_provider()` in `factory.py`. Then `create_llm_model()` can use IDs such as:

```python
create_llm_model(
    model_id="openai:gpt-4.1",
    ...
)
```

Update the factory's client arguments and validation to support the new SDK client.

## Design Rules

- Keep `LLMModel` and `LLMProvider` as `Protocol` interfaces.
- Use `BaseLLMModel` for shared implementation.
- Keep provider-specific API logic inside that provider's model class.
- Keep shared schemas independent of provider SDKs.
- Add tests for text generation, streaming, tool calls, errors, and timeouts.
