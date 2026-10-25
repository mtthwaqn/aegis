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
        if _surrounded_by_sky(frame, x, y, w, h):
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


def _surrounded_by_sky(frame: NDArray[np.uint8], x: int, y: int, w: int, h: int) -> bool:
    """A bright blob in white sky is a cloud, not a person in the water."""
    height, width = frame.shape[:2]
    pad = max(6, min(w, h))
    y0, x0 = max(0, y - pad), max(0, x - pad)
    y1, x1 = min(height, y + h + pad), min(width, x + w + pad)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    ring = np.ones(hsv.shape[:2], dtype=bool)
    ring[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = False
    context = hsv[ring]
    if context.size < 60:
        return False
    hue = float(context[:, 0].mean())
    saturation = float(context[:, 1].mean())
    value = float(context[:, 2].mean())
    # Open water in the swimmer fixture sits near value 150. Sky is much brighter.
    if value > 185 and saturation < 80:
        return True
    return value > 190 and 95 <= hue <= 130 and saturation > 40


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
