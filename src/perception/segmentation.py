"""Separate compact objects from the open-water background."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

REFERENCE_WIDTH = 1920


@dataclass(frozen=True, slots=True)
class Component:
    x: int
    y: int
    w: int
    h: int
    area: int
    contrast: float

    def gap_to(self, other: Component) -> float:
        gap_x = _gap(self.x, self.w, other.x, other.w)
        gap_y = _gap(self.y, self.h, other.y, other.h)
        return float(np.hypot(gap_x, gap_y))

    def union(self, other: Component) -> Component:
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return Component(
            x,
            y,
            w,
            h,
            self.area + other.area,
            max(self.contrast, other.contrast),
        )


def _gap(origin: int, size: int, other_origin: int, other_size: int) -> float:
    if origin + size < other_origin:
        return float(other_origin - (origin + size))
    if other_origin + other_size < origin:
        return float(origin - (other_origin + other_size))
    return 0.0


def scale_of(width: int) -> float:
    return max(width, 1) / REFERENCE_WIDTH


def foreground(frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Mask pixels that differ from a coarse water background.

    The background is a strong downsample, so a swimmer disappears into it and
    shows up again in the difference image. Sun glitter stays, and is rejected
    later by contrast and size.
    """
    height, width = frame.shape[:2]
    small_w = max(8, width // 24)
    small_h = max(8, height // 24)
    background = cv2.resize(
        cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    score = cv2.GaussianBlur(
        cv2.cvtColor(cv2.absdiff(frame, background), cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )
    _, mask = cv2.threshold(score, 18, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    close = max(3, int(round(15 * scale_of(width))))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return closed, score


def components(mask: NDArray[np.uint8], score: NDArray[np.uint8]) -> list[Component]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    scale = scale_of(mask.shape[1])
    min_area = 150 * scale * scale
    found: list[Component] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < min_area or w < 1 or h < 1:
            continue
        contrast = float(score[labels == index].mean())
        found.append(Component(x, y, w, h, area, contrast))
    return found
