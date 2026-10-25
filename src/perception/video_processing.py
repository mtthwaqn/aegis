"""Run one frame through the perception pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from src.perception.hazard_detection import detect_hazards, overlaps, recover_vessels
from src.perception.preprocessing import low_visibility, preprocess
from src.perception.segmentation import components, foreground
from src.perception.target_detection import Detection, detect_targets
from src.perception.trained_detector import detect_trained, use_trained_detector
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
    """Detect on one frame.

    Classical detection is the default. ``AEGIS_DETECTOR=trained`` selects the
    person/boat weights at confidence 0.45. Weights on disk do not switch this.
    """
    started = time.perf_counter()
    if use_trained_detector():
        return _analyze_trained(frame, started)
    return _analyze_classical(frame, started)


def _analyze_classical(frame: NDArray[np.uint8], started: float) -> PerceptionResult:
    height, width = frame.shape[:2]
    working = frame
    if width > 640:
        working = cv2.resize(frame, (640, round(height * 640 / width)))
    processed = preprocess(working)
    mask, score = foreground(processed)
    regions = components(mask, score)
    # Color checks use the resized camera frame. CLAHE makes clouds look like objects.
    hazards = detect_hazards(working, recover_vessels(processed, score, regions), [])
    people = detect_targets(
        [region for region in regions if not overlaps(region, hazards)],
        working,
    )
    detections = people + hazards
    annotated = annotate(processed, detections)
    if working is not frame:
        factor = width / working.shape[1]
        detections = [_scale_detection(item, factor) for item in detections]
    elapsed = (time.perf_counter() - started) * 1000
    return PerceptionResult(
        annotated=annotated,
        detections=detections,
        latency_ms=elapsed,
        is_occluded=low_visibility(processed) and not people,
        width=width,
        height=height,
    )


def _scale_detection(detection: Detection, factor: float) -> Detection:
    x, y, w, h = detection.coordinates
    return Detection(
        classification=detection.classification,
        coordinates=(x * factor, y * factor, w * factor, h * factor),
        confidence=detection.confidence,
        label=detection.label,
    )


def _analyze_trained(frame: NDArray[np.uint8], started: float) -> PerceptionResult:
    processed = preprocess(frame)
    height, width = frame.shape[:2]
    detections = detect_trained(frame)
    people = [item for item in detections if item.classification == "survivor"]
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
    """Pack detections into the shared vision contract. Geographic fields stay empty."""
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
