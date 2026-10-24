"""Call the vision tools, then one action tool, and keep the execution trace.

The trace is a record of tool calls. It is not evidence that hardware moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.actions.decide import Decision, decide
from src.mcp.tools import assess_frame, read_vision, reanalyze_frame
from src.server.schemas import AgentActionPayload, VisionState


@dataclass(frozen=True, slots=True)
class TraceStep:
    tool: str
    detail: str


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    frame_id: int
    outcome: str
    reason: str
    steps: tuple[TraceStep, ...]
    action: AgentActionPayload | None
    hardware_executed: bool = False


def run_agent(
    vision: VisionState,
    frame: NDArray[np.uint8] | None = None,
    *,
    threshold: float = 0.65,
) -> ExecutionTrace:
    """Read vision, re-analyze degraded frames once, then dispatch, reroute, or escalate."""
    steps: list[TraceStep] = []
    reading = read_vision(vision, threshold)
    steps.append(TraceStep("read_vision", str(reading["state"])))
    blurred = False
    if frame is not None:
        quality = assess_frame(frame)
        blurred = quality["blurred"]
        steps.append(TraceStep("assess_frame", "blur" if blurred else "sharp"))

    first = decide(vision, blurred=blurred, threshold=threshold)
    chosen = first
    if frame is not None and first.outcome == "escalate_to_human":
        second_vision = reanalyze_frame(frame, vision.frame_id, vision.timestamp)
        steps.append(TraceStep("reanalyze_frame", "live detector"))
        second_reading = read_vision(second_vision, threshold)
        steps.append(TraceStep("read_vision", str(second_reading["state"])))
        # A second look may confirm a fix. It must not clear blur or occlusion.
        if not blurred and not vision.is_occluded:
            second = decide(second_vision, blurred=False, threshold=threshold)
            if second.outcome in {"dispatch_payload", "reroute_ground_rescue"}:
                chosen = second

    if chosen.action is not None:
        steps.append(TraceStep(chosen.outcome, chosen.reason))
        action = chosen.action.model_copy(update={"reasoning_trace": _text(steps, chosen.reason)})
    else:
        action = None
    return ExecutionTrace(
        frame_id=vision.frame_id,
        outcome=chosen.outcome,
        reason=chosen.reason,
        steps=tuple(steps),
        action=action,
        hardware_executed=False,
    )


def trace_payload(trace: ExecutionTrace) -> dict[str, Any]:
    return {
        "frame_id": trace.frame_id,
        "outcome": trace.outcome,
        "reason": trace.reason,
        "hardware_executed": trace.hardware_executed,
        "steps": [{"tool": step.tool, "detail": step.detail} for step in trace.steps],
        "action": None if trace.action is None else trace.action.model_dump(mode="json"),
    }


def _text(steps: list[TraceStep], reason: str) -> str:
    chain = " → ".join(step.tool for step in steps)
    return f"{reason} {chain}"
