"""Validated environment configuration; secrets never appear in telemetry."""

from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    video_source: str = "mock"
    ingestion_fps: Annotated[float, Field(gt=0, le=240)] = 30.0
    broadcast_hz: Annotated[float, Field(ge=1, le=30)] = 25.0
    mock_actions: bool = True
    api_token: Annotated[str, Field(max_length=1024)] = ""
    allowed_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    compute_instance: str = "AWS Graviton3 c7g.xlarge + COOL"
    deployment_verified: bool = False
    hourly_compute_cost_usd: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    low_confidence_threshold: Annotated[float, Field(gt=0, le=1)] = 0.65
    frame_queue_size: Annotated[int, Field(ge=1, le=32)] = 2
    frame_cache_size: Annotated[int, Field(ge=2, le=300)] = 120
    jpeg_quality: Annotated[int, Field(ge=20, le=95)] = 80
    vision_wait_ms: Annotated[float, Field(ge=0, le=2000)] = 150.0
    source_stale_seconds: Annotated[float, Field(gt=0)] = 5.0
    read_timeout_ms: Annotated[int, Field(ge=100, le=10000)] = 3000
    kinesis_region: str = "ap-southeast-1"
    kinesis_data_endpoint: str | None = None
    kinesis_session_seconds: Annotated[int, Field(ge=300, le=43200)] = 3600

    @model_validator(mode="after")
    def finite_rates(self) -> "Settings":
        import math

        for value in (
            self.ingestion_fps,
            self.broadcast_hz,
            self.vision_wait_ms,
            self.source_stale_seconds,
        ):
            if not math.isfinite(value):
                raise ValueError("Timing configuration must be finite")
        return self
