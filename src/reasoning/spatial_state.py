from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel

from src.server.schemas import Coordinates, VisionState

from .confidence import ConfidenceStatus, assess_confidence


class SpatialState(str, Enum):
    TARGET_VERIFIED = "TARGET_VERIFIED"
    HAZARD_DETECTED = "HAZARD_DETECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_TARGET = "NO_TARGET"


PixelToCoordinates = Callable[[VisionState, float, float], Coordinates | dict[str, float] | None]


def evaluate_spatial_state(
    vision: VisionState,
    low_confidence_threshold: float = 0.65,
) -> SpatialState:
    assess_confidence(1.0, threshold=low_confidence_threshold)
    assessments = [
        assess_confidence(
            box.confidence,
            threshold=low_confidence_threshold,
            is_occluded=vision.is_occluded,
        )
        for box in vision.bounding_boxes
    ]
    if vision.is_occluded or any(
        assessment.status is ConfidenceStatus.LOW_CONFIDENCE for assessment in assessments
    ):
        return SpatialState.LOW_CONFIDENCE
    if any(
        box.classification in {"hazard", "debris"}
        for box in vision.bounding_boxes
    ):
        return SpatialState.HAZARD_DETECTED
    if any(box.classification == "survivor" for box in vision.bounding_boxes):
        return SpatialState.TARGET_VERIFIED
    return SpatialState.NO_TARGET


def georeference_vision_state(
    vision: VisionState,
    pixel_to_coordinates: PixelToCoordinates | None = None,
    drone_coordinates: Coordinates | dict[str, float] | None = None,
) -> VisionState:
    updated_boxes = []
    for box in vision.bounding_boxes:
        coordinates = box.geo_coordinates
        if coordinates is None and pixel_to_coordinates is not None:
            cx, cy = box.centroid
            coordinates = pixel_to_coordinates(vision, cx, cy)
            if coordinates is not None:
                coordinates = _validated_coordinates(coordinates)
        updated_boxes.append(
            box.model_copy(update={"geo_coordinates": coordinates}).model_dump(mode="python")
        )
    payload = vision.model_dump(mode="python")
    payload["bounding_boxes"] = updated_boxes
    if drone_coordinates is not None:
        payload["drone_coordinates"] = _validated_coordinates(drone_coordinates)
    return VisionState.model_validate(payload)


def _validated_coordinates(value: Coordinates | dict[str, float]) -> Coordinates:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    return Coordinates.model_validate(value)
