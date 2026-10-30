from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .domain import AgentPolicy, BudgetLimits


class BudgetConfig(BaseModel):
    max_steps: int = Field(gt=0)
    max_seconds: float = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_cost_usd: float = Field(gt=0)

    def to_limits(self) -> BudgetLimits:
        """
        Convert the configuration into a BudgetLimits instance.

        Returns:
            BudgetLimits: Budget limits used by the agent graph.
        """
        return BudgetLimits(
            max_steps=self.max_steps,
            max_seconds=self.max_seconds,
            max_tokens=self.max_tokens,
            max_cost_usd=self.max_cost_usd,
        )


class AgentConfig(BaseModel):
    policy: AgentPolicy = Field(default_factory=AgentPolicy)
    budget: BudgetConfig


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
