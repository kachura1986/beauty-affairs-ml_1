from pydantic import BaseModel, ConfigDict, Field


class AgentPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    empty_limit: int = Field(default=2, gt=0)
    repeat_limit: int = Field(default=3, gt=0)