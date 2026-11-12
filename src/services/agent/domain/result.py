from pydantic import BaseModel, ConfigDict, Field

from .reason import Finalization, TerminationReason


class AgentResult(BaseModel):
    """Stable result of a completed agent_graph run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(description="Run identifier; matches the trace file stem.")
    agent_name: str | None = Field(
        default=None,
        description="Agent that produced this result (None for ad-hoc runs).",
    )
    reason: TerminationReason
    answer: str | None = None
    steps: int
    tokens: int
    cost: float
    seconds: float
    history_len: int
    trace: str
    finalization: Finalization = Field(
        default=Finalization.NONE,
        description=(
            "How the final answer was produced. LLM — synthesized by the model  in finalize_node;"
            "FALLBACK — deterministic fallback from the last tool result; NONE — no finalization was attempted."
        ),
    )
