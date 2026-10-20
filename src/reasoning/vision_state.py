from __future__ import annotations

from typing import Any

from src.server.schemas import BoundingBox, Coordinates, VisionState

from .spatial_state import SpatialState, evaluate_spatial_state


def to_reasoning_payload(
    vision: VisionState,
    low_confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    state = evaluate_spatial_state(vision, low_confidence_threshold)
    indexed_boxes = list(enumerate(vision.bounding_boxes, start=1))
    survivors = [(index, box) for index, box in indexed_boxes if box.classification == "survivor"]
    hazards = [
        (index, box)
        for index, box in indexed_boxes
        if box.classification in {"hazard", "debris"}
    ]
    target = _detection_payload(vision.frame_id, *max(survivors, key=_confidence, default=(0, None)))
    hazard_detections = [
        _detection_payload(vision.frame_id, index, box) for index, box in hazards
    ]
    hazard_detection = max(hazards, key=_confidence, default=(0, None))[1]
    selected_detection = _location_detection(state, survivors, hazards)
    confidence_score = min((box.confidence for box in vision.bounding_boxes), default=None)
    if state is SpatialState.LOW_CONFIDENCE:
        confidence_status = "LOW_CONFIDENCE"
    elif confidence_score is None:
        confidence_status = "NO_DETECTION"
    else:
        confidence_status = "HIGH_CONFIDENCE"

    return {
        "frame_id": vision.frame_id,
        "timestamp": vision.timestamp.isoformat(),
        "frame_width": vision.frame_width,
        "frame_height": vision.frame_height,
        "latency_ms": vision.latency_ms,
        "state": state.value,
        "target": target,
        "hazard": {
            "detected": bool(hazards),
            "type": hazard_detection.classification if hazard_detection else None,
            "confidence": hazard_detection.confidence if hazard_detection else None,
            "detections": hazard_detections,
        },
        "location": _coordinate_payload(
            selected_detection.geo_coordinates if selected_detection else None
        ),
        "drone_coordinates": _coordinate_payload(vision.drone_coordinates),
        "confidence": {"score": confidence_score, "status": confidence_status},
        "is_occluded": vision.is_occluded,
        "bounding_boxes": [box.model_dump(mode="json") for box in vision.bounding_boxes],
    }


def _detection_payload(
    frame_id: int,
    index: int,
    box: BoundingBox | None,
) -> dict[str, Any] | None:
    if box is None:
        return None
    return {
        "id": f"F{frame_id}-B{index}",
        "id_scope": "frame",
        "type": "person" if box.classification == "survivor" else box.classification,
        "classification": box.classification,
        "confidence": box.confidence,
        "bounding_box": box.coordinates,
        "centroid": box.centroid,
        "geo_coordinates": _coordinate_payload(box.geo_coordinates),
    }


def _location_detection(
    state: SpatialState,
    survivors: list[tuple[int, BoundingBox]],
    hazards: list[tuple[int, BoundingBox]],
) -> BoundingBox | None:
    if state is SpatialState.HAZARD_DETECTED:
        return max(hazards, key=_confidence, default=(0, None))[1]
    if state in {SpatialState.TARGET_VERIFIED, SpatialState.LOW_CONFIDENCE}:
        candidates = survivors + hazards
        return max(candidates, key=_confidence, default=(0, None))[1]
    return None


def _confidence(item: tuple[int, BoundingBox]) -> float:
    return item[1].confidence


def _coordinate_payload(coordinates: Coordinates | None) -> dict[str, float] | None:
    if coordinates is None:
        return None
    return coordinates.model_dump(mode="json")
