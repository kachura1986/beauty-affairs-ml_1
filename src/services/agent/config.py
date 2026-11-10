from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .domain import AgentPolicy, BudgetLimits

DEFAULT_AGENT_KEY = 'default_agent'


class ModelSettings(BaseModel):
    """Per-agent LLM model settings."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(description="Model name used by this agent.")
    temperature: float = Field(
        ge=0, le=2,
        description="Sampling temperature for this agent's model.",
    )
    max_tokens: int = Field(
        gt=0,
        description=(
            "Per-call max_output_tokens for a single LLM call. "
            "NOT the run-wide BudgetLimits.max_tokens — that one caps the whole run."
        ),
    )
    input_cost_per_million: float = Field(ge=0)
    output_cost_per_million: float = Field(ge=0)


class AgentConfig(BaseModel):
    model: ModelSettings
    policy: AgentPolicy
    budget: BudgetLimits


def load_agent_configs(path: Path) -> dict[str, AgentConfig]:
    """
    Load and validate agent configurations from a YAML file.

    Args:
        path (Path): Path to the agents YAML file.

    Returns:
        dict[str, AgentConfig]: Mapping of agent name to its validated configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        pydantic.ValidationError: If the file contents do not match the schema.
    """
    with path.open(encoding='utf-8') as fh:
        raw = yaml.safe_load(fh) or {}

    return {name: AgentConfig.model_validate(config) for name, config in raw.items()}


def require_default_agent(agent_configs: dict[str, AgentConfig], source: Path) -> AgentConfig:
    """Return the ``default_agent`` config, or raise ``ValueError`` naming the source file.

    CONTRACT: the composition root requires ``default_agent`` to exist. A missing key must fail with a message that
    names the file and the keys actually present, so a startup failure is diagnosable from logs alone.

    Args:
        agent_configs (dict[str, AgentConfig]): Mapping returned by :func:`load_agent_configs`.
        source (Path): Path to the agents YAML file, included in the error message.

    Returns:
        AgentConfig: The ``default_agent`` configuration.

    Raises:
        ValueError: If ``agent_configs`` has no ``'default_agent'`` key.
    """
    agent_config = agent_configs.get(DEFAULT_AGENT_KEY)

    if agent_config is None:
        raise ValueError(f"{source}: must contain '{DEFAULT_AGENT_KEY}'; got keys: {sorted(agent_configs)}")

    return agent_config
