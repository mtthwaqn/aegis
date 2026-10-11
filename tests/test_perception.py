from datetime import UTC, datetime

import cv2
import numpy as np

from src.perception.video_processing import analyze, to_vision_state


def _sea(width: int = 960, height: int = 540) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (150, 145, 80)
    noise = np.random.default_rng(0).integers(0, 6, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def test_open_water_has_no_person() -> None:
    result = analyze(_sea())
    assert result.detections == []
    assert result.width == 960
    assert result.height == 540


def test_bright_swimmer_becomes_contract_box() -> None:
    frame = _sea()
    cv2.ellipse(frame, (480, 250), (24, 9), 15, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(frame, (458, 244), (10, 4), 15, 0, 360, (30, 30, 40), -1)
    result = analyze(frame)
    people = [item for item in result.detections if item.classification == "survivor"]
    assert len(people) == 1
    person = people[0]
    x, y, w, h = person.coordinates
    assert x <= 480 <= x + w
    assert y <= 250 <= y + h
    state = to_vision_state(result, frame_id=3, timestamp=datetime.now(UTC))
    assert state.frame_id == 3
    assert state.bounding_boxes[0].classification == "survivor"
    assert state.drone_coordinates is None


def test_large_obstacle_is_hazard_and_warm_fragment_is_debris() -> None:
    frame = _sea()
    cv2.rectangle(frame, (80, 80), (230, 150), (20, 20, 20), -1)
    cv2.rectangle(frame, (700, 400), (732, 422), (40, 80, 220), -1)
    kinds = {item.classification for item in analyze(frame).detections}
    assert "hazard" in kinds
    assert "debris" in kinds
