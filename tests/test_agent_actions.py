import asyncio
import base64
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from src.agent.runner import run_agent
from src.ingestion.video_streamer import VideoStreamer
from src.mcp.server import TOOLS, handle_message
from src.mcp.tools import dispatch_payload
from src.perception.preprocessing import is_blurred
from src.perception.video_processing import analyze, to_vision_state
from src.server.bus import IntegrationBus
from src.server.config import Settings
from src.server.schemas import BoundingBox, Coordinates, VisionState
from tests.test_perception import _sea


def _box(kind: str, confidence: float, geo: Coordinates | None = None) -> BoundingBox:
    return BoundingBox(
        coordinates=[10.0, 10.0, 20.0, 20.0],
        centroid=[20.0, 20.0],
        confidence=confidence,
        classification=kind,
        geo_coordinates=geo,
    )


def _vision(
    boxes: list[BoundingBox] | None = None,
    *,
    occluded: bool = False,
    frame_id: int = 7,
    drone: Coordinates | None = None,
) -> VisionState:
    return VisionState(
        frame_id=frame_id,
        timestamp=datetime.now(UTC),
        latency_ms=12.0,
        bounding_boxes=boxes or [],
        is_occluded=occluded,
        frame_width=960,
        frame_height=540,
        drone_coordinates=drone,
    )


def _place() -> Coordinates:
    return Coordinates(lat=14.6, lon=120.98)


def _detailed() -> np.ndarray:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[::2, ::2] = (255, 255, 255)
    image[1::2, 1::2] = (20, 20, 20)
    return image


def _blurred() -> np.ndarray:
    return cv2.GaussianBlur(_sea(), (0, 0), 11)


def _names(trace) -> list[str]:
    return [step.tool for step in trace.steps]


def test_target_verified_dispatches_payload() -> None:
    trace = run_agent(_vision([_box("survivor", 0.91, _place())]))
    assert trace.outcome == "dispatch_payload"
    assert trace.hardware_executed is False
    assert trace.action is not None
    assert trace.action.action_type == "DISPATCH_SUPPLY_DRONE"
    assert trace.action.payload_type == "rescue_payload"
    assert trace.action.target_coordinates == _place()
    assert "Dispatch payload." in trace.action.reasoning_trace
    assert _names(trace) == ["read_vision", "dispatch_payload"]


def test_hazard_reroutes_ground_rescue() -> None:
    drone = Coordinates(lat=14.59, lon=120.97)
    trace = run_agent(_vision([_box("hazard", 0.88, _place())], drone=drone))
    assert trace.outcome == "reroute_ground_rescue"
    assert trace.action is not None
    assert trace.action.action_type == "REROUTE_CONVOY"
    assert trace.action.payload_type == "ground_rescue"
    assert trace.action.route_coordinates == [drone, _place()]
    assert "Reroute ground rescue." in trace.reason
    assert "reroute_ground_rescue" in _names(trace)


def test_low_confidence_escalates_without_reanalysis() -> None:
    trace = run_agent(_vision([_box("survivor", 0.4, _place())]))
    assert trace.outcome == "escalate_to_human"
    assert trace.action is not None
    assert trace.action.action_type == "ESCALATE_HUMAN"
    assert "Low confidence." in trace.reason
    assert "reanalyze_frame" not in _names(trace)


def test_no_target_holds() -> None:
    trace = run_agent(_vision())
    assert trace.outcome == "hold"
    assert trace.action is None
    assert trace.reason == "No target."
    assert trace.hardware_executed is False
    assert _names(trace) == ["read_vision"]


def test_blur_escalates_after_reanalysis() -> None:
    frame = _blurred()
    assert is_blurred(frame)
    assert not is_blurred(_detailed())
    trace = run_agent(_vision([_box("survivor", 0.95, _place())]), frame)
    assert trace.outcome == "escalate_to_human"
    assert "Frame blur." in trace.reason
    assert _names(trace) == [
        "read_vision",
        "assess_frame",
        "reanalyze_frame",
        "read_vision",
        "escalate_to_human",
    ]


def test_occlusion_escalates_and_is_not_called_blur() -> None:
    frame = _detailed()
    assert not is_blurred(frame)
    trace = run_agent(_vision([_box("survivor", 0.95, _place())], occluded=True), frame)
    assert trace.outcome == "escalate_to_human"
    assert "Frame occlusion." in trace.reason
    assert "Frame blur." not in trace.reason
    assert "reanalyze_frame" in _names(trace)
    assert "escalate_to_human" in _names(trace)


def test_reanalysis_does_not_clear_blur(monkeypatch: pytest.MonkeyPatch) -> None:
    confirmed = _vision([_box("survivor", 0.95, _place())])
    monkeypatch.setattr("src.agent.runner.reanalyze_frame", lambda *_args, **_kwargs: confirmed)
    trace = run_agent(_vision(), _blurred())
    assert trace.outcome == "escalate_to_human"
    assert "Frame blur." in trace.reason


def test_reanalysis_can_confirm_a_geographic_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    confirmed = _vision([_box("survivor", 0.95, _place())])
    monkeypatch.setattr("src.agent.runner.reanalyze_frame", lambda *_args, **_kwargs: confirmed)
    trace = run_agent(_vision([_box("survivor", 0.42)]), _detailed())
    assert trace.outcome == "dispatch_payload"
    assert "reanalyze_frame" in _names(trace)
    assert _names(trace)[-1] == "dispatch_payload"


def test_verified_target_without_a_fix_does_not_dispatch() -> None:
    trace = run_agent(_vision([_box("survivor", 0.91)]))
    assert trace.outcome == "escalate_to_human"
    assert "No geographic fix." in trace.reason
    assert trace.action is not None
    assert trace.action.target_coordinates == Coordinates(lat=0.0, lon=0.0)


def test_origin_zero_is_not_a_drop_point() -> None:
    nowhere = Coordinates(lat=0.0, lon=0.0)
    trace = run_agent(_vision([_box("survivor", 0.91, nowhere)]))
    assert trace.outcome == "escalate_to_human"


def test_classical_detection_does_not_invent_a_dispatch() -> None:
    frame = _sea()
    cv2.ellipse(frame, (480, 250), (18, 7), 15, 0, 360, (235, 235, 235), -1)
    state = to_vision_state(analyze(frame), frame_id=3, timestamp=datetime.now(UTC))
    assert state.bounding_boxes
    assert all(box.geo_coordinates is None for box in state.bounding_boxes)
    trace = run_agent(state, frame)
    assert trace.hardware_executed is False
    assert trace.outcome != "dispatch_payload"


def test_mcp_lists_vision_and_action_tools() -> None:
    response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [tool["name"] for tool in response["result"]["tools"]]
    assert names == [tool["name"] for tool in TOOLS]
    assert names == [
        "read_vision",
        "assess_frame",
        "reanalyze_frame",
        "dispatch_payload",
        "reroute_ground_rescue",
        "escalate_to_human",
    ]


def test_mcp_read_vision_and_refuses_a_mismatched_action() -> None:
    vision = _vision([_box("survivor", 0.91, _place())])
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "read_vision",
                "arguments": {"vision": vision.model_dump(mode="json")},
            },
        }
    )
    assert response["result"]["isError"] is False
    assert "TARGET_VERIFIED" in response["result"]["content"][0]["text"]
    with pytest.raises(ValueError, match="dispatch_payload"):
        dispatch_payload(_vision())


def test_mcp_reanalyze_roundtrip() -> None:
    encoded = base64.b64encode(cv2.imencode(".jpg", _detailed())[1].tobytes()).decode("ascii")
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "reanalyze_frame",
                "arguments": {
                    "image_base64": encoded,
                    "frame_id": 4,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            },
        }
    )
    assert response["result"]["isError"] is False
    assert '"frame_id": 4' in response["result"]["content"][0]["text"]


async def test_agent_trace_is_recorded_on_the_bus(settings: Settings) -> None:
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    try:
        trace = run_agent(_vision([_box("survivor", 0.91, _place())]))
        assert trace.action is not None
        recorded = await bus.record_action(trace.action)
        assert recorded.origin == "agent"
        assert recorded.status == "accepted"
        assert "dispatch_payload" in recorded.action.reasoning_trace
        assert recorded.stages[-1] == "action_dispatched"
    finally:
        await streamer.close()
