from pathlib import Path

import pytest
from pydantic import ValidationError

from services.agent.config import (
    DEFAULT_AGENT_KEY,
    AgentConfig,
    ModelSettings,
    load_agent_configs,
    require_default_agent,
)
from services.agent.domain import AgentPolicy, BudgetLimits


def _model_settings() -> ModelSettings:
    """Build a minimal valid ModelSettings."""
    return ModelSettings(
        model_name='gpt-4o-mini',
        temperature=0,
        max_tokens=8192,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    )


def _agent_config() -> AgentConfig:
    """Build a minimal valid AgentConfig; the payload itself is irrelevant to the helper."""
    return AgentConfig(
        model=_model_settings(),
        policy=AgentPolicy(empty_limit=3, repeat_limit=3),
        budget=BudgetLimits(max_steps=5, max_seconds=60, max_tokens=10_000, max_cost_usd=0),
    )


# ---------------------------------------------------------------------------
# require_default_agent: startup validation contract
# ---------------------------------------------------------------------------

def test_require_default_agent_returns_config():
    """A present 'default_agent' key is returned unchanged (identity, not a copy)."""
    cfg = _agent_config()
    source = Path('/etc/agents.yaml')

    result = require_default_agent({'default_agent': cfg}, source)

    assert result is cfg


def test_require_default_agent_empty_mapping_names_source_and_keys():
    """An empty mapping fails with the source path and an empty key list."""
    source = Path('/etc/agents.yaml')

    with pytest.raises(ValueError) as exc_info:
        require_default_agent({}, source)

    message = str(exc_info.value)
    assert str(source) in message
    assert "must contain 'default_agent'" in message
    assert "got keys: []" in message


def test_require_default_agent_missing_key_lists_available_keys_sorted():
    """A missing key fails with the available keys in deterministic order."""
    cfg = _agent_config()
    source = Path('/etc/agents.yaml')

    with pytest.raises(ValueError) as exc_info:
        require_default_agent({'cheap_agent': cfg, 'another_agent': cfg}, source)

    message = str(exc_info.value)
    assert "must contain 'default_agent'" in message
    assert "got keys: ['another_agent', 'cheap_agent']" in message  # sorted


# ---------------------------------------------------------------------------
# DEFAULT_AGENT_KEY: single source of truth for the public-agent identifier
# ---------------------------------------------------------------------------

def test_default_agent_key_is_stable():
    """The key is shared by require_default_agent and get_agent_service — it must not drift."""
    assert DEFAULT_AGENT_KEY == 'default_agent'


# ---------------------------------------------------------------------------
# AgentConfig.model: mandatory per-agent LLM settings
# ---------------------------------------------------------------------------

def test_agent_config_requires_model_block():
    """model is intentionally mandatory: a forgotten block fails at startup, not at first request."""
    cfg = {
        'policy': AgentPolicy(empty_limit=3, repeat_limit=3).model_dump(),
        'budget': BudgetLimits(
            max_steps=5, max_seconds=60, max_tokens=10_000, max_cost_usd=0,
        ).model_dump(),
    }

    with pytest.raises(ValidationError, match='model'):
        AgentConfig.model_validate(cfg)


# ---------------------------------------------------------------------------
# ModelSettings: bounds and immutability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('field,value', [
    ('temperature', 3),  # > le=2
    ('temperature', -0.1),  # < ge=0
    ('max_tokens', 0),  # > gt=0
    ('input_cost_per_million', -1),
    ('output_cost_per_million', -1),
])
def test_model_settings_bounds(field: str, value: float | int):
    """Out-of-range values are rejected at validation time (fail-fast on startup)."""
    bad = {
        'model_name': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 8192,
        'input_cost_per_million': 0.15,
        'output_cost_per_million': 0.60,
    }
    bad[field] = value

    with pytest.raises(ValidationError):
        ModelSettings.model_validate(bad)


def test_model_settings_is_frozen():
    """ModelSettings is frozen: assignment raises ValidationError (requires pydantic>=2)."""
    settings = _model_settings()

    with pytest.raises(ValidationError):
        settings.temperature = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_agent_configs: YAML round-trip
# ---------------------------------------------------------------------------

_YAML_DOCUMENT = """
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
"""


def test_load_agent_configs_round_trip(tmp_path: Path):
    """YAML→AgentConfig round-trip; per-call and run-wide token budgets stay distinct."""
    path = tmp_path / 'agents.yaml'
    path.write_text(_YAML_DOCUMENT)

    configs = load_agent_configs(path)

    assert list(configs) == ['default_agent']
    cfg = configs['default_agent']
    assert cfg.model.model_name == 'gpt-4o-mini'
    assert cfg.model.max_tokens == 8192  # per-call max_output_tokens
    assert cfg.budget.max_tokens == 100_000  # run-wide token budget
    assert cfg.model.max_tokens != cfg.budget.max_tokens  # not the same field
