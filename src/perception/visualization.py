"""Draw detection boxes on a copy of the frame."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.perception.target_detection import Detection

_COLORS = {
    "survivor": (80, 220, 120),
    "hazard": (40, 60, 220),
    "debris": (40, 180, 230),
}


def annotate(frame: NDArray[np.uint8], detections: list[Detection]) -> NDArray[np.uint8]:
    canvas = frame.copy()
    for detection in detections:
        x, y, w, h = (int(round(value)) for value in detection.coordinates)
        color = _COLORS[detection.classification]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        origin = (x, max(18, y - 6))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas
