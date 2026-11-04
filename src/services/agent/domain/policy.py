from pydantic import BaseModel, ConfigDict, Field


class AgentPolicy(BaseModel):
    """
    Agent limit policy.

    Defines upper bounds for abnormal-situation counters so the agent cannot loop indefinitely
    on source failures or repeated retries.

    Attributes:
        empty_limit (int): Maximum consecutive empty responses. Range: [1, 5].
        repeat_limit (int): Maximum retry attempts on repetition. Range: [1, 10].
        finalize_on_max_steps (bool): When True and ``max_steps`` is exhausted, the graph makes one additional
            LLM call with tools disabled to synthesize a final answer from the collected history, subject to
            the remaining wall-clock and token budgets. This call is the only place that may bypass ``Budget.can_start_step()``.
    """

    model_config = ConfigDict(frozen=True)

    empty_limit: int = Field(gt=0, le=5, description="Maximum consecutive empty responses.")
    repeat_limit: int = Field(gt=0, le=10, description="Maximum retry attempts on repetition.")
    finalize_on_max_steps: bool = Field(
        default=True,
        description=(
            "When True and max_steps is exhausted, the graph makes one additional LLM call with tools disabled"
            "to synthesize a final answer from the collected history, subject to remaining_seconds() and"
            "remaining_tokens(). This call is the only place that may bypass can_start_step()."
        ),
    )
