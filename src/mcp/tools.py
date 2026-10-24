"""Vision and action tools. Action tools record intent and do not move hardware."""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.actions.decide import Decision, decide
from src.perception.preprocessing import is_blurred
from src.perception.video_processing import analyze, to_vision_state
from src.reasoning.vision_state import to_reasoning_payload
from src.server.schemas import VisionState

HARDWARE_EXECUTED = False


def read_vision(vision: VisionState, threshold: float = 0.65) -> dict[str, Any]:
    """Return the reasoning view of one VisionState."""
    return to_reasoning_payload(vision, threshold)


def assess_frame(frame: NDArray[np.uint8]) -> dict[str, bool]:
    """Report blur. Occlusion stays on the VisionState flag, not this measurement."""
    return {"blurred": is_blurred(frame)}


def reanalyze_frame(frame: NDArray[np.uint8], frame_id: int, timestamp: datetime) -> VisionState:
    """Run the live detector again. The live default remains the classical detector."""
    return to_vision_state(analyze(frame), frame_id, timestamp)


def dispatch_payload(vision: VisionState, threshold: float = 0.65) -> dict[str, Any]:
    return _tool_result(decide(vision, threshold=threshold), "dispatch_payload")


def reroute_ground_rescue(vision: VisionState, threshold: float = 0.65) -> dict[str, Any]:
    return _tool_result(decide(vision, threshold=threshold), "reroute_ground_rescue")


def escalate_to_human(
    vision: VisionState, *, blurred: bool = False, threshold: float = 0.65
) -> dict[str, Any]:
    return _tool_result(decide(vision, blurred=blurred, threshold=threshold), "escalate_to_human")


def decode_jpeg(image_base64: str) -> NDArray[np.uint8]:
    raw = base64.b64decode(image_base64, validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("image_base64 is not a JPEG frame")
    return frame


def _tool_result(decision: Decision, expected: str) -> dict[str, Any]:
    if decision.outcome != expected or decision.action is None:
        raise ValueError(f"{expected} is not the decision for this VisionState")
    return {
        "hardware_executed": HARDWARE_EXECUTED,
        "outcome": decision.outcome,
        "reason": decision.reason,
        "action": decision.action.model_dump(mode="json"),
    }
