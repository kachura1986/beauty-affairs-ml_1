from pydantic import BaseModel, ConfigDict, Field


class AgentPolicy(BaseModel):
    """
    Agent limit policy.

    Defines upper bounds for abnormal-situation counters so the agent cannot loop indefinitely
    on source failures or repeated retries.

    Attributes:
        empty_limit (int): Maximum consecutive empty responses. Range: [1, 5].
        repeat_limit (int): Maximum retry attempts on repetition. Range: [1, 10].
    """

    model_config = ConfigDict(frozen=True)

    empty_limit: int = Field(gt=0, le=5, description="Maximum consecutive empty responses.")
    repeat_limit: int = Field(gt=0, le=10, description="Maximum retry attempts on repetition.")
