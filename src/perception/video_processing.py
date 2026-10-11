"""Run one frame through the perception pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from src.perception.hazard_detection import detect_hazards, overlaps
from src.perception.preprocessing import low_visibility, preprocess
from src.perception.segmentation import components, foreground
from src.perception.target_detection import Detection, detect_targets
from src.perception.visualization import annotate
from src.server.schemas import BoundingBox, VisionState


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    annotated: NDArray[np.uint8]
    detections: list[Detection]
    latency_ms: float
    is_occluded: bool
    width: int
    height: int


def analyze(frame: NDArray[np.uint8]) -> PerceptionResult:
    started = time.perf_counter()
    processed = preprocess(frame)
    mask, score = foreground(processed)
    regions = components(mask, score)
    height, width = processed.shape[:2]
    hazards = detect_hazards(processed, regions, [])
    people = detect_targets(
        [region for region in regions if not overlaps(region, hazards)],
        width,
        height,
    )
    detections = people + hazards
    elapsed = (time.perf_counter() - started) * 1000
    return PerceptionResult(
        annotated=annotate(processed, detections),
        detections=detections,
        latency_ms=elapsed,
        is_occluded=low_visibility(processed) and not people,
        width=width,
        height=height,
    )


def to_vision_state(result: PerceptionResult, frame_id: int, timestamp: datetime) -> VisionState:
    """Pack detections into Member 1's contract. Geographic fields stay empty."""
    boxes = [
        BoundingBox(
            coordinates=list(detection.coordinates),
            centroid=list(detection.centroid),
            confidence=detection.confidence,
            classification=detection.classification,
        )
        for detection in result.detections
    ]
    return VisionState(
        frame_id=frame_id,
        timestamp=timestamp,
        latency_ms=result.latency_ms,
        bounding_boxes=boxes,
        is_occluded=result.is_occluded,
        frame_width=result.width,
        frame_height=result.height,
    )
