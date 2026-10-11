"""Find people in the water from compact high-contrast regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.perception.segmentation import Component, scale_of

Classification = Literal["survivor", "hazard", "debris"]
MIN_CONTRAST = 31.0
MIN_PERSON_AREA = 800
MAX_PERSON_AREA = 4500
MIN_BOX_BRIGHTNESS = 100.0


@dataclass(frozen=True, slots=True)
class Detection:
    classification: Classification
    coordinates: tuple[float, float, float, float]
    confidence: float
    label: str

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.coordinates
        return (x + w / 2, y + h / 2)


def detect_targets(
    regions: list[Component],
    frame: NDArray[np.uint8],
) -> list[Detection]:
    """Keep person-sized blobs and join nearby body parts into one box."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = scale_of(width)
    people = [region for region in regions if _person_shaped(region, scale)]
    merged = _merge_nearby(people, max_gap=40 * scale)
    pad = max(1, int(round(8 * scale)))
    detections: list[Detection] = []
    for region in merged:
        if not _person_shaped(region, scale):
            continue
        x = max(0, region.x - pad)
        y = max(0, region.y - pad)
        w = min(width - x, region.w + 2 * pad)
        h = min(height - y, region.h + 2 * pad)
        if float(gray[y : y + h, x : x + w].mean()) < MIN_BOX_BRIGHTNESS:
            continue
        confidence = min(0.92, max(0.45, region.contrast / 60))
        detections.append(
            Detection(
                classification="survivor",
                coordinates=(float(x), float(y), float(w), float(h)),
                confidence=confidence,
                label="PERSON",
            )
        )
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def _person_shaped(region: Component, scale: float) -> bool:
    area_scale = scale * scale
    if region.contrast < MIN_CONTRAST:
        return False
    if not (MIN_PERSON_AREA * area_scale) <= region.area <= (MAX_PERSON_AREA * area_scale):
        return False
    short = min(region.w, region.h)
    long = max(region.w, region.h)
    if short < 10 * scale or long / short > 5:
        return False
    return region.area / (region.w * region.h) >= 0.2


def _merge_nearby(regions: list[Component], max_gap: float) -> list[Component]:
    pending = list(regions)
    merged: list[Component] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remaining: list[Component] = []
            for other in pending:
                if current.gap_to(other) <= max_gap:
                    current = current.union(other)
                    changed = True
                else:
                    remaining.append(other)
            pending = remaining
        merged.append(current)
    return merged
