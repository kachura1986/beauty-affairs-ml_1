import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependencies import PipelineDependencies, get_agent_service
from services.agent.config import DEFAULT_AGENT_KEY

MULTI_AGENT_YAML = """
default_agent:
  model:
    model_name: gpt-4o-mini
    temperature: 0
    max_tokens: 8192
    input_cost_per_million: 0.15
    output_cost_per_million: 0.60
  policy:
    empty_limit: 2
    repeat_limit: 3
    finalize_on_max_steps: true
  budget:
    max_steps: 12
    max_seconds: 120
    max_tokens: 100000
    max_cost_usd: 0.5
classifier_agent:
  model:
    model_name: gpt-4o-mini
    temperature: 0
    max_tokens: 256
    input_cost_per_million: 0.15
    output_cost_per_million: 0.60
  policy:
    empty_limit: 2
    repeat_limit: 3
    finalize_on_max_steps: true
  budget:
    max_steps: 3
    max_seconds: 30
    max_tokens: 5000
    max_cost_usd: 0.1
"""

SINGLE_AGENT_YAML = """
only_agent:
  model:
    model_name: gpt-4o-mini
    temperature: 0
    max_tokens: 8192
    input_cost_per_million: 0.15
    output_cost_per_million: 0.60
  policy:
    empty_limit: 2
    repeat_limit: 3
    finalize_on_max_steps: true
  budget:
    max_steps: 12
    max_seconds: 120
    max_tokens: 100000
    max_cost_usd: 0.5
"""

MIN_TOKENS_AGENT_YAML = """
default_agent:
  model:
    model_name: gpt-4o-mini
    temperature: 0
    max_tokens: 8192
    input_cost_per_million: 0.15
    output_cost_per_million: 0.60
  policy:
    empty_limit: 2
    repeat_limit: 3
    finalize_on_max_steps: true
  budget:
    max_steps: 12
    max_seconds: 120
    max_tokens: 1
    max_cost_usd: 0.5
"""


@pytest.fixture
def agents_config(tmp_path: Path, monkeypatch):
    """Write a YAML at tmp_path and point settings.AGENTS_CONFIG at it; return a setter."""
    from core.config import settings

    def _set(text: str) -> Path:
        path = tmp_path / 'agents.yaml'
        path.write_text(text)
        monkeypatch.setattr(settings, 'AGENTS_CONFIG', path)
        return path

    return _set


@pytest.fixture
def patched_externals():
    """Replace every heavy external client used by create(); yield a namespace of mocks.

    CONTRACT: openai_client and search_index_client are returned individually so
    failure-path tests can assert ``.close().await_count``. Both are MagicMock with
    ``close = AsyncMock()`` — otherwise ``await client.close()`` in the
    ``except BaseException`` branch of create() would raise.
    """

    def _mock_async_client():
        client = MagicMock()
        client.close = AsyncMock()
        return client

    openai_client = _mock_async_client()
    search_index_client = _mock_async_client()

    with (
        patch('dependencies.AsyncOpenAI', return_value=openai_client),
        patch('dependencies.SearchIndexClient', return_value=search_index_client),
        patch('dependencies.AzureKeyCredential'),
        patch('dependencies.AzureOpenAIEmbedder'),
        patch('dependencies.AzureSearchIndexManager', return_value=_mock_async_client()),
        patch('dependencies.ProductsSearchIndex'),
        patch('dependencies.create_product_tools', return_value=MagicMock()),
        patch('dependencies.OpenAICompatibleModel') as mock_model,
        patch('dependencies.IndexService'),
    ):
        yield SimpleNamespace(
            model=mock_model,
            openai_client=openai_client,
            search_index_client=search_index_client,
        )


async def test_create_builds_registry_per_agent(agents_config, patched_externals):
    """One AzureOpenAIModel and one AgentService per configured agent."""
    agents_config(MULTI_AGENT_YAML)

    deps = await PipelineDependencies.create()

    try:
        assert isinstance(deps.agent_services, dict)
        assert set(deps.agent_services) == {'default_agent', 'classifier_agent'}

        # One AzureOpenAIModel per agent, with per-agent kwargs.
        assert patched_externals.model.call_count == 2
        max_tokens = {
            call.kwargs['max_tokens'] for call in patched_externals.model.call_args_list
        }
        assert max_tokens == {8192, 256}
    finally:
        await deps.close()


async def test_get_agent_service_returns_default_agent(agents_config, patched_externals):
    """get_agent_service returns the service registered under DEFAULT_AGENT_KEY."""
    agents_config(MULTI_AGENT_YAML)
    deps = await PipelineDependencies.create()

    try:
        request = MagicMock()
        request.app.state.dependencies = deps

        assert get_agent_service(request) is deps.agent_services[DEFAULT_AGENT_KEY]
    finally:
        await deps.close()


async def test_create_fails_when_default_agent_missing(agents_config, patched_externals):
    """Startup must fail fast, not on the first /api/agent/chat request.

    Also verifies that partially-created clients are closed — the except
    BaseException branch of create() must not leak openai_client or
    search_index_client.
    """
    agents_config(SINGLE_AGENT_YAML)

    with pytest.raises(ValueError, match='default_agent'):
        await PipelineDependencies.create()

    assert patched_externals.openai_client.close.await_count == 1
    assert patched_externals.search_index_client.close.await_count == 1


async def test_create_fails_when_budget_tokens_below_min_expected(agents_config, patched_externals):
    """Startup must fail fast when budget.max_tokens is below MIN_EXPECTED_TOKENS.

    Otherwise the agent would boot, then die on the first LLM step with BUDGET_TOKENS
    and no diagnosable message — the pre-flight token check rejects every call.

    Also verifies that partially-created clients are closed — the except
    BaseException branch of create() must not leak openai_client or
    search_index_client (same contract as test_create_fails_when_default_agent_missing).

    ``max_tokens=1`` (not 100) keeps the test green regardless of future
    MIN_EXPECTED_TOKENS values: any positive default is > 1, so the fail-fast
    branch is always reached.
    """
    agents_config(MIN_TOKENS_AGENT_YAML)

    with pytest.raises(ValueError, match='MIN_EXPECTED_TOKENS'):
        await PipelineDependencies.create()

    assert patched_externals.openai_client.close.await_count == 1
    assert patched_externals.search_index_client.close.await_count == 1


async def test_create_logs_agent_services(agents_config, patched_externals, caplog):
    """Startup log must name every created agent service (diagnostics)."""
    agents_config(MULTI_AGENT_YAML)

    with caplog.at_level(logging.INFO):
        deps = await PipelineDependencies.create()
        await deps.close()

    assert any(
        'Created 2 agent service(s)' in r.message and 'classifier_agent' in r.message
        for r in caplog.records
    )
