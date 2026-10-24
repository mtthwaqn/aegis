"""Person and boat boxes from the trained detector.

Classical detection stays active unless ``AEGIS_DETECTOR=trained``. A weights
file on disk does not switch the live path.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.perception.target_detection import Classification, Detection
from src.perception.trained_classes import CLASSIFICATION_OF

LIVE_CONFIDENCE = 0.45
DEFAULT_WEIGHTS = Path("data/training/runs/person-boat-sds/weights/best.pt")
MAX_BOXES = 500
_LABEL = {"person": "PERSON", "boat": "BOAT"}

_model: object | None = None
_loaded: Path | None = None


def use_trained_detector() -> bool:
    mode = os.environ.get("AEGIS_DETECTOR", "classical").strip().lower()
    if mode in {"", "classical"}:
        return False
    if mode == "trained":
        return True
    raise ValueError("AEGIS_DETECTOR must be 'classical' or 'trained'")


def live_confidence() -> float:
    raw = os.environ.get("AEGIS_DETECTOR_CONF", str(LIVE_CONFIDENCE))
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("AEGIS_DETECTOR_CONF must be a number from 0 to 1") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("AEGIS_DETECTOR_CONF must be a number from 0 to 1")
    return value


def weights_path() -> Path:
    return Path(os.environ.get("AEGIS_DETECTOR_WEIGHTS", str(DEFAULT_WEIGHTS)))


def boxes_to_detections(
    names: dict[int, str],
    xyxy: list[list[float]],
    classes: list[int],
    confidences: list[float],
    width: int,
    height: int,
    min_confidence: float,
) -> list[Detection]:
    """Map model boxes into the vision contract. Unknown classes are dropped."""
    detections: list[Detection] = []
    for corners, class_id, confidence in zip(xyxy, classes, confidences, strict=True):
        name = names.get(int(class_id), "")
        kind = CLASSIFICATION_OF.get(name)
        if kind is None or float(confidence) < min_confidence:
            continue
        clipped = _clip(corners, width, height)
        if clipped is None:
            continue
        classification: Classification = "survivor" if kind == "survivor" else "hazard"
        detections.append(
            Detection(
                classification=classification,
                coordinates=clipped,
                confidence=float(confidence),
                label=_LABEL[name],
            )
        )
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections[:MAX_BOXES]


def detect_trained(frame: NDArray[np.uint8]) -> list[Detection]:
    """Run the person/boat weights on the original frame, before contrast stretch."""
    height, width = frame.shape[:2]
    confidence = live_confidence()
    predicted = _load(weights_path()).predict(
        frame,
        conf=confidence,
        imgsz=640,
        verbose=False,
        device=inference_device(),
    )
    result = predicted[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = {int(key): str(value) for key, value in result.names.items()}
    return boxes_to_detections(
        names,
        _as_rows(boxes.xyxy),
        [int(value) for value in _as_list(boxes.cls)],
        [float(value) for value in _as_list(boxes.conf)],
        width,
        height,
        confidence,
    )


def _clip(
    corners: list[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(value) for value in corners)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    return (x1, y1, box_w, box_h)


def _load(path: Path) -> object:
    global _model, _loaded
    resolved = path.resolve()
    if _model is not None and _loaded == resolved:
        return _model
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained weights not found at {path}. "
            "Set AEGIS_DETECTOR_WEIGHTS, or leave AEGIS_DETECTOR unset."
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Trained detection needs the optional ultralytics package: pip install ultralytics"
        ) from error
    _model = YOLO(str(path))
    _loaded = resolved
    return _model


def inference_device() -> str:
    """Prefer CUDA, then Apple MPS. CPU covers Windows and Graviton."""
    try:
        import torch
    except ImportError:
        return "cpu"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _as_list(values: object) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()  # type: ignore[union-attr]
    return values.tolist()  # type: ignore[union-attr]


def _as_rows(values: object) -> list[list[float]]:
    rows = _as_list(values)
    return [list(row) for row in rows]
