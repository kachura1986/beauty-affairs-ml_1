from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .domain import AgentPolicy, BudgetLimits


class AgentConfig(BaseModel):
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
