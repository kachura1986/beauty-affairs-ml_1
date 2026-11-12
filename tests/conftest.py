import os

# OpenAI-compatible transport
os.environ.setdefault("OPENAI_BASE_URL", "https://test.openai.example/v1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Azure search
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_MODEL", "embed-test")
os.environ.setdefault("SEARCH_API_KEY", "test-search-key")
os.environ.setdefault("SEARCH_ENDPOINT", "https://test.search.windows.net")
