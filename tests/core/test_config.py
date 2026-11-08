from core.config import Settings

REMOVED_FIELDS = (
    'AZURE_OPENAI_CHAT_MODEL',
    'TEMPERATURE',
    'MAX_TOKENS',
    'INPUT_COST_PER_MILLION',
    'OUTPUT_COST_PER_MILLION',
)

RETAINED_FIELDS = (
    'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_API_KEY',
    'AZURE_OPENAI_EMBEDDING_MODEL',
    'EMBEDDING_DIMENSIONS',
)


def test_model_settings_removed_from_settings():
    """Single source of truth: model settings live per-agent in AgentConfig.model."""
    fields = Settings.model_fields
    for name in REMOVED_FIELDS:
        assert name not in fields, f"{name} must not exist in Settings"


def test_transport_and_embedding_fields_stay_global():
    """Endpoint/key are per-resource (shared by chat + embedding); embeddings are shared."""
    fields = Settings.model_fields
    for name in RETAINED_FIELDS:
        assert name in fields, f"{name} must still exist in Settings"
