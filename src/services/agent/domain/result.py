from pydantic import BaseModel, ConfigDict

from .reason import TerminationReason


class AgentResult(BaseModel):
    """Stable result of a completed agent_graph run."""

    model_config = ConfigDict(frozen=True)

    reason: TerminationReason
    answer: str | None = None
    steps: int
    tokens: int
    cost_usd: float
    seconds: float
    history_len: int
    trace: str
