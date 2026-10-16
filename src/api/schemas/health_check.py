from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the application")
    app: str = Field(..., description="Name of the application")
