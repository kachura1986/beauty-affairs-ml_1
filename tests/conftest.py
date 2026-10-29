import os

os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
os.environ.setdefault("AZURE_OPENAI_CHAT_MODEL", "gpt-test")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_MODEL", "embed-test")
os.environ.setdefault("SEARCH_API_KEY", "test-search-key")
os.environ.setdefault("SEARCH_ENDPOINT", "https://test.search.windows.net")