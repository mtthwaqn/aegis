import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.server.schemas import BoundingBox, Coordinates, VisionState


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_id": "1"},
        {"frame_id": True},
        {"is_occluded": "false"},
        {"latency_ms": -1.0},
        {"latency_ms": float("nan")},
        {"unknown": "field"},
        {"timestamp": datetime(2026, 1, 1).isoformat()},
        {"bounding_boxes": "[]"},
    ],
)
def test_invalid_vision_rejected(vision: VisionState, changes: dict[str, object]) -> None:
    payload = vision.model_dump(mode="json") | changes
    with pytest.raises(ValidationError):
        VisionState.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181), (float("inf"), 0), (True, 0)])
def test_invalid_geo_rejected(lat: float, lon: float) -> None:
    with pytest.raises(ValidationError):
        Coordinates(lat=lat, lon=lon)


def test_centroid_and_bounds_checked(vision: VisionState) -> None:
    with pytest.raises(ValidationError):
        BoundingBox(
            coordinates=[0.0, 0.0, 20.0, 20.0],
            centroid=[1.0, 1.0],
            confidence=0.9,
            classification="survivor",
        )
    box = BoundingBox(
        coordinates=[950.0, 0.0, 20.0, 20.0],
        centroid=[960.0, 10.0],
        confidence=0.9,
        classification="survivor",
    )
    with pytest.raises(ValidationError):
        VisionState.model_validate(vision.model_dump() | {"bounding_boxes": [box]})


def test_json_roundtrip(vision: VisionState) -> None:
    assert VisionState.model_validate_json(vision.model_dump_json()) == vision
