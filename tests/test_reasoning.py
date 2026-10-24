from pathlib import Path

import pytest

from src.reasoning.vision_state import to_reasoning_payload
from src.server.schemas import BoundingBox, Coordinates, ReasoningState, VisionState


@pytest.mark.parametrize(
    "sample,state,confidence,hazard",
    [
        ("survivor", "TARGET_VERIFIED", "HIGH_CONFIDENCE", False),
        ("hazard", "LOW_CONFIDENCE", "LOW_CONFIDENCE", True),
        ("empty", "NO_TARGET", "NO_DETECTION", False),
    ],
)
def test_team_samples_have_typed_reasoning(
    sample: str, state: str, confidence: str, hazard: bool
) -> None:
    path = Path(__file__).resolve().parents[1] / "samples" / "vision" / f"{sample}.json"
    vision = VisionState.model_validate_json(path.read_text())
    reasoning = ReasoningState.model_validate(to_reasoning_payload(vision))
    assert reasoning.frame_id == vision.frame_id
    assert reasoning.state == state
    assert reasoning.confidence.status == confidence
    assert reasoning.hazard.detected == hazard
    assert reasoning.location is None
    assert (reasoning.target is not None) == (sample == "survivor")
    assert ReasoningState.model_validate_json(reasoning.model_dump_json()) == reasoning


def test_state_priorities_confidence_and_selected_location(vision: VisionState) -> None:
    person = BoundingBox(
        coordinates=[10.0, 10.0, 20.0, 20.0],
        centroid=[20.0, 20.0],
        classification="survivor",
        confidence=0.9,
        geo_coordinates=Coordinates(lat=14.6, lon=120.98),
    )
    hazard = BoundingBox(
        coordinates=[40.0, 40.0, 20.0, 20.0],
        centroid=[50.0, 50.0],
        classification="hazard",
        confidence=0.7,
        geo_coordinates=Coordinates(lat=14.7, lon=120.99),
    )
    state = vision.model_copy(update={"bounding_boxes": [person, hazard]})
    result = ReasoningState.model_validate(to_reasoning_payload(state))
    assert result.target.confidence == 0.9
    assert result.confidence.score == 0.7
    assert result.state == "HAZARD_DETECTED"
    assert result.location == hazard.geo_coordinates
    occluded = state.model_copy(update={"is_occluded": True})
    result = ReasoningState.model_validate(to_reasoning_payload(occluded))
    assert result.state == "LOW_CONFIDENCE"
    assert result.hazard.detected  # Low confidence must not hide a detected hazard.
    result = ReasoningState.model_validate(to_reasoning_payload(state, 0.8))
    assert result.state == "LOW_CONFIDENCE"


def test_drone_gps_is_not_substituted_for_missing_detection_location(vision: VisionState) -> None:
    state = vision.model_copy(update={"drone_coordinates": Coordinates(lat=14.6, lon=120.98)})
    reasoning = ReasoningState.model_validate(to_reasoning_payload(state))
    assert reasoning.location is None
    assert reasoning.drone_coordinates == state.drone_coordinates
