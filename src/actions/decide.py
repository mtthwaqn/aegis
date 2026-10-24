"""Map one VisionState onto dispatch, ground-rescue reroute, or human review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from src.reasoning.spatial_state import SpatialState, evaluate_spatial_state
from src.server.schemas import AgentActionPayload, BoundingBox, Coordinates, VisionState

Outcome = Literal["dispatch_payload", "reroute_ground_rescue", "escalate_to_human", "hold"]


@dataclass(frozen=True, slots=True)
class Decision:
    outcome: Outcome
    reason: str
    action: AgentActionPayload | None


def decide(vision: VisionState, *, blurred: bool = False, threshold: float = 0.65) -> Decision:
    """Choose at most one action. Missing or 0,0 coordinates are not dispatch targets."""
    state = evaluate_spatial_state(vision, threshold)
    if blurred:
        return _escalate(vision, "Escalate to human. Frame blur.")
    if vision.is_occluded:
        return _escalate(vision, "Escalate to human. Frame occlusion.")
    if state is SpatialState.LOW_CONFIDENCE:
        return _escalate(vision, "Escalate to human. Low confidence.")
    if state is SpatialState.HAZARD_DETECTED:
        hazard = _best(vision, {"hazard", "debris"})
        if hazard is None or not _usable(hazard.geo_coordinates):
            return _escalate(vision, "Escalate to human. No geographic fix.")
        return _reroute(vision, hazard.geo_coordinates)
    if state is SpatialState.TARGET_VERIFIED:
        target = _best(vision, {"survivor"})
        if target is None or not _usable(target.geo_coordinates):
            return _escalate(vision, "Escalate to human. No geographic fix.")
        return _dispatch(vision, target.geo_coordinates)
    return Decision("hold", "No target.", None)


def _best(vision: VisionState, kinds: set[str]) -> BoundingBox | None:
    boxes = [box for box in vision.bounding_boxes if box.classification in kinds]
    if not boxes:
        return None
    return max(boxes, key=lambda box: box.confidence)


def _usable(coordinates: Coordinates | None) -> bool:
    return coordinates is not None and not (coordinates.lat == 0.0 and coordinates.lon == 0.0)


def _target(vision: VisionState) -> Coordinates:
    for box in vision.bounding_boxes:
        if _usable(box.geo_coordinates):
            return box.geo_coordinates
    if _usable(vision.drone_coordinates):
        return vision.drone_coordinates
    return Coordinates(lat=0.0, lon=0.0)


def _dispatch(vision: VisionState, coordinates: Coordinates) -> Decision:
    reason = "Dispatch payload."
    return Decision(
        "dispatch_payload",
        reason,
        _action(vision, "DISPATCH_SUPPLY_DRONE", "rescue_payload", coordinates, reason, "dispatch_payload"),
    )


def _reroute(vision: VisionState, coordinates: Coordinates) -> Decision:
    reason = "Reroute ground rescue."
    route: list[Coordinates] = []
    if _usable(vision.drone_coordinates):
        route = [vision.drone_coordinates, coordinates]
    return Decision(
        "reroute_ground_rescue",
        reason,
        _action(
            vision,
            "REROUTE_CONVOY",
            "ground_rescue",
            coordinates,
            reason,
            "reroute_ground_rescue",
            route,
        ),
    )


def _escalate(vision: VisionState, reason: str) -> Decision:
    return Decision(
        "escalate_to_human",
        reason,
        _action(
            vision,
            "ESCALATE_HUMAN",
            "operator_review",
            _target(vision),
            reason,
            "escalate_to_human",
        ),
    )


def _action(
    vision: VisionState,
    action_type: Literal["DISPATCH_SUPPLY_DRONE", "REROUTE_CONVOY", "ESCALATE_HUMAN"],
    payload_type: str,
    coordinates: Coordinates,
    reason: str,
    outcome: str,
    route: list[Coordinates] | None = None,
) -> AgentActionPayload:
    return AgentActionPayload(
        action_id=f"agent-{vision.frame_id}-{outcome}",
        action_type=action_type,
        target_coordinates=coordinates,
        payload_type=payload_type,
        reasoning_trace=reason,
        source_frame_id=vision.frame_id,
        timestamp=datetime.now(UTC),
        route_coordinates=route or [],
    )
