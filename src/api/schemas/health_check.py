from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the application")
    app: str = Field(..., description="Name of the application")


class ProviderStatus(BaseModel):
    status: str = Field(..., description="Status of the provider")
    latency_ms: int = Field(..., description="Provider response latency in milliseconds")


class ProviderHealth(BaseModel):
    status: str = Field(..., description="Overall status of configured providers")
    providers: dict[str, ProviderStatus] = Field(
        ...,
        description="Status information for each configured provider",
    )
