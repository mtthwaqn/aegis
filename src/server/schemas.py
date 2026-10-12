"""Versioned, strict JSON contracts shared by every Aegis team member."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegative = Annotated[Finite, Field(ge=0)]
Confidence = Annotated[Finite, Field(ge=0, le=1)]
ActionType = Literal["DISPATCH_SUPPLY_DRONE", "REROUTE_CONVOY", "ESCALATE_HUMAN"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Coordinates(Contract):
    lat: Annotated[Finite, Field(ge=-90, le=90)]
    lon: Annotated[Finite, Field(ge=-180, le=180)]


class BoundingBox(Contract):
    """Pixel coordinates in the original frame, never normalized or geographic."""

    coordinates: Annotated[list[NonNegative], Field(min_length=4, max_length=4)]
    centroid: Annotated[list[NonNegative], Field(min_length=2, max_length=2)]
    confidence: Confidence
    classification: Literal["survivor", "hazard", "debris"]
    geo_coordinates: Coordinates | None = None

    @model_validator(mode="after")
    def consistent_geometry(self) -> Self:
        x, y, w, h = self.coordinates
        if w <= 0 or h <= 0:
            raise ValueError("Bounding box width and height must be positive")
        if not all(
            math.isclose(a, b, abs_tol=0.5)
            for a, b in zip(self.centroid, (x + w / 2, y + h / 2), strict=True)
        ):
            raise ValueError("Centroid must match the bounding box within half a pixel")
        return self


class VisionState(Contract):
    frame_id: Annotated[int, Field(ge=0)]
    timestamp: AwareDatetime = Field(strict=False)
    latency_ms: NonNegative
    bounding_boxes: Annotated[list[BoundingBox], Field(max_length=500)]
    is_occluded: bool
    frame_width: Annotated[int, Field(ge=1, le=16384)]
    frame_height: Annotated[int, Field(ge=1, le=16384)]
    drone_coordinates: Coordinates | None = None

    @model_validator(mode="after")
    def boxes_fit_frame(self) -> Self:
        for box in self.bounding_boxes:
            x, y, w, h = box.coordinates
            if x + w > self.frame_width or y + h > self.frame_height:
                raise ValueError("Bounding box extends outside the original frame")
        return self


class AgentActionPayload(Contract):
    action_id: Annotated[str, Field(min_length=1, max_length=128)] = Field(
        default_factory=lambda: str(uuid4())
    )
    action_type: ActionType
    target_coordinates: Coordinates
    payload_type: Annotated[str, Field(min_length=1, max_length=128)]
    reasoning_trace: Annotated[str, Field(min_length=1, max_length=8192)]
    source_frame_id: Annotated[int, Field(ge=0)] | None = None
    timestamp: AwareDatetime = Field(strict=False)
    route_coordinates: Annotated[list[Coordinates], Field(max_length=128)] = Field(
        default_factory=list
    )


class DecisionTrace(Contract):
    sequence: Annotated[int, Field(ge=1)]
    action: AgentActionPayload
    received_at: AwareDatetime = Field(strict=False)
    status: Literal["accepted", "blocked", "escalated"]
    origin: Literal["agent", "safety", "mock"]
    stages: list[str]


class Alert(Contract):
    alert_id: str
    severity: Literal["info", "warning", "critical"]
    message: str
    frame_id: int | None = None


class OverrideRequest(Contract):
    enabled: bool
    reason: Annotated[str, Field(min_length=1, max_length=512)]


class ControlState(Contract):
    override_enabled: bool
    reason: str
    updated_at: AwareDatetime = Field(strict=False)


class SystemTelemetry(Contract):
    schema_version: Literal["1.0"] = "1.0"
    timestamp: AwareDatetime = Field(strict=False)
    active_fps: NonNegative
    vision_fps: NonNegative
    p95_latency_ms: NonNegative
    cloud_compute_instance: str
    deployment_verified: bool
    estimated_cost_usd_per_100k_frames: NonNegative | None
    fleet_status: Literal["simulation", "awaiting_agent", "agent_connected", "human_control"]
    active_alerts: list[Alert]
    action_log: Annotated[list[DecisionTrace], Field(max_length=50)]
    control: ControlState
    source_status: Literal["starting", "streaming", "reconnecting", "stopped", "failed"]
    dropped_frames: Annotated[int, Field(ge=0)]
    connected_viewers: Annotated[int, Field(ge=0)]
    metrics_window_seconds: NonNegative
    telemetry_age_ms: NonNegative | None


class VideoFramePayload(Contract):
    frame_id: Annotated[int, Field(ge=0)]
    timestamp: AwareDatetime = Field(strict=False)
    width: int
    height: int
    jpeg_base64: str


class LiveEnvelope(Contract):
    type: Literal["live"] = "live"
    sequence: int
    frame: VideoFramePayload | None
    vision: VisionState | None
    telemetry: SystemTelemetry
