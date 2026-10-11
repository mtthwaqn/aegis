"""Find non-person objects. Open water alone is not boxed as a flood."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.perception.segmentation import Component, scale_of
from src.perception.target_detection import MAX_PERSON_AREA, MIN_CONTRAST, Detection


def detect_hazards(
    frame: NDArray[np.uint8],
    regions: list[Component],
    people: list[Detection],
) -> list[Detection]:
    """Boats and similar large obstacles become hazards. Warm fragments become debris.

    A person floating in open water does not create a second hazard box. Flood
    severity is not a MOBDrone label, and a guessed water polygon would not be
    a real detection.
    """
    scale = scale_of(frame.shape[1])
    area_scale = scale * scale
    detections: list[Detection] = []
    for region in regions:
        if region.contrast < MIN_CONTRAST or overlaps(region, people):
            continue
        kind = _classify(frame, region, area_scale, scale)
        if kind is None:
            continue
        classification, label, confidence = kind
        detections.append(
            Detection(
                classification=classification,
                coordinates=(float(region.x), float(region.y), float(region.w), float(region.h)),
                confidence=confidence,
                label=label,
            )
        )
    return detections


def _classify(
    frame: NDArray[np.uint8],
    region: Component,
    area_scale: float,
    scale: float,
) -> tuple[str, str, float] | None:
    short = min(region.w, region.h)
    long = max(region.w, region.h)
    if short < 8 * scale or long / short > 6:
        return None
    confidence = min(0.85, max(0.4, region.contrast / 70))
    center_x = region.x + region.w // 2
    center_y = region.y + region.h // 2
    half = 3
    patch = frame[
        max(0, center_y - half) : center_y + half + 1,
        max(0, center_x - half) : center_x + half + 1,
    ]
    blue, green, red = cv2.mean(patch)[:3]
    warm = red > 140 and red > green + 15 and red > blue + 30
    if warm and region.area <= (MAX_PERSON_AREA * area_scale):
        return ("debris", "DEBRIS", confidence)
    if (MAX_PERSON_AREA * area_scale) < region.area <= (90000 * area_scale):
        return ("hazard", "HAZARD", confidence)
    return None


def overlaps(region: Component, detections: list[Detection]) -> bool:
    box = np.array([region.x, region.y, region.x + region.w, region.y + region.h], dtype=float)
    for detection in detections:
        x, y, w, h = detection.coordinates
        other = np.array([x, y, x + w, y + h], dtype=float)
        ix = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
        iy = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
        if ix * iy > 0:
            return True
    return False
