"""Tests for require_default_agent: startup validation contract."""
from pathlib import Path

import pytest

from services.agent.config import AgentConfig, require_default_agent
from services.agent.domain import AgentPolicy, BudgetLimits


def _agent_config() -> AgentConfig:
    """Build a minimal valid AgentConfig; the payload itself is irrelevant to the helper."""
    return AgentConfig(
        policy=AgentPolicy(empty_limit=3, repeat_limit=3),
        budget=BudgetLimits(max_steps=5, max_seconds=60, max_tokens=10_000, max_cost_usd=0),
    )


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
