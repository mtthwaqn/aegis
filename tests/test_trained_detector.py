from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.perception.trained_detector import boxes_to_detections
from src.perception.video_processing import analyze, to_vision_state
from tests.test_perception import _sea


def test_boxes_keep_person_and_boat_above_cutoff() -> None:
    detections = boxes_to_detections(
        {0: "person", 1: "boat", 2: "buoy"},
        [[10, 20, 30, 50], [1, 1, 11, 11], [40, 40, 80, 60], [0, 0, 4, 4]],
        [0, 2, 1, 1],
        [0.90, 0.99, 0.70, 0.20],
        width=200,
        height=200,
        min_confidence=0.45,
    )
    assert [(item.label, item.classification) for item in detections] == [
        ("PERSON", "survivor"),
        ("BOAT", "hazard"),
    ]
    assert detections[0].coordinates == (10.0, 20.0, 20.0, 30.0)
    assert detections[0].confidence == 0.90


def test_boxes_clip_to_the_frame() -> None:
    detections = boxes_to_detections(
        {0: "person"},
        [[-5, -8, 15, 12]],
        [0],
        [0.50],
        width=100,
        height=80,
        min_confidence=0.45,
    )
    assert detections[0].coordinates == (0.0, 0.0, 15.0, 12.0)


def test_analyze_stays_classical_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGIS_DETECTOR", raising=False)

    def fail(frame: np.ndarray) -> list[object]:
        raise AssertionError("trained detector ran")

    monkeypatch.setattr("src.perception.video_processing.detect_trained", fail)
    result = analyze(_sea())
    assert result.detections == []


def test_trained_flag_publishes_model_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_DETECTOR", "trained")
    monkeypatch.setattr(
        "src.perception.video_processing.detect_trained",
        lambda frame: boxes_to_detections(
            {0: "person", 1: "boat"},
            [[10, 12, 30, 28], [40, 50, 90, 80]],
            [0, 1],
            [0.80, 0.60],
            frame.shape[1],
            frame.shape[0],
            0.45,
        ),
    )
    result = analyze(_sea())
    state = to_vision_state(result, frame_id=7, timestamp=datetime.now(UTC))
    assert [box.classification for box in state.bounding_boxes] == ["survivor", "hazard"]
    assert state.drone_coordinates is None
    assert all(box.geo_coordinates is None for box in state.bounding_boxes)


def test_inference_device_is_not_forced_to_mps() -> None:
    from src.perception.trained_detector import inference_device

    assert inference_device() in {"cpu", "cuda", "mps"}


def test_training_scripts_do_not_force_mps() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("train_detector.py", "eval_trained.py"):
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert 'device="mps"' not in text
        assert "device='mps'" not in text


def test_unknown_detector_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_DETECTOR", "yolo")
    with pytest.raises(ValueError, match="AEGIS_DETECTOR"):
        analyze(_sea())
