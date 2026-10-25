from datetime import UTC, datetime

import cv2
import numpy as np

from src.perception.video_processing import analyze, to_vision_state


def _sea(width: int = 960, height: int = 540) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (150, 145, 80)
    noise = np.random.default_rng(0).integers(0, 6, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def test_blurred_frame_is_occluded() -> None:
    frame = cv2.GaussianBlur(_sea(), (0, 0), 11)
    result = analyze(frame)
    assert result.is_occluded
    assert result.detections == []
    state = to_vision_state(result, frame_id=1, timestamp=datetime.now(UTC))
    assert state.bounding_boxes == []


def test_open_water_has_no_person() -> None:
    result = analyze(_sea())
    assert result.detections == []
    assert result.width == 960
    assert result.height == 540


def test_bright_swimmer_becomes_contract_box() -> None:
    frame = _sea()
    cv2.ellipse(frame, (480, 250), (18, 7), 15, 0, 360, (235, 235, 235), -1)
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


def test_a_cloud_is_not_a_person() -> None:
    for background, axes in (((220, 170, 120), (18, 12)), ((235, 220, 200), (22, 14))):
        frame = np.full((540, 960, 3), background, dtype=np.uint8)
        cv2.ellipse(frame, (480, 250), axes, 0, 0, 360, (255, 255, 255), -1)
        people = [item for item in analyze(frame).detections if item.classification == "survivor"]
        assert people == []


def test_a_merged_scene_keeps_the_bright_vessel() -> None:
    from src.perception.hazard_detection import recover_vessels
    from src.perception.segmentation import Component

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[:] = (80, 90, 70)
    frame[40:90, 300:360] = (230, 230, 230)
    score = np.zeros((200, 400), dtype=np.uint8)
    score[20:180, 20:380] = 30
    score[40:90, 300:360] = 80
    giant = Component(20, 20, 360, 160, 50_000, 40.0)
    recovered = recover_vessels(frame, score, [giant])
    assert any(piece.x >= 280 for piece in recovered)
    assert all(piece.w < 200 for piece in recovered)


def test_large_obstacle_is_hazard_and_warm_fragment_is_debris() -> None:
    frame = _sea()
    cv2.rectangle(frame, (80, 80), (230, 150), (20, 20, 20), -1)
    cv2.rectangle(frame, (700, 400), (724, 414), (40, 80, 220), -1)
    kinds = {item.classification for item in analyze(frame).detections}
    assert "hazard" in kinds
    assert "debris" in kinds
