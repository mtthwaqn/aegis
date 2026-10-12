import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from src.ingestion.video_streamer import CapturedFrame, VideoStreamer
from src.server.bus import BusConflict, IntegrationBus, StoredFrame
from src.server.config import Settings
from src.server.schemas import (
    AgentActionPayload,
    BoundingBox,
    LiveEnvelope,
    OverrideRequest,
    VideoFramePayload,
    VisionState,
)


def insert(bus: IntegrationBus, frame_id: int = 0) -> VideoFramePayload:
    payload = VideoFramePayload(
        frame_id=frame_id, timestamp=datetime.now(UTC), width=960, height=540, jpeg_base64=""
    )
    bus.frames[frame_id] = StoredFrame(payload, b"", time.monotonic())
    return payload


async def test_ring_idempotency_and_override(
    settings: Settings, action: AgentActionPayload
) -> None:
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    try:
        first = await bus.record_action(action)
        assert await bus.record_action(action) == first
        with pytest.raises(BusConflict):
            await bus.record_action(action.model_copy(update={"payload_type": "different"}))
        for i in range(65):
            await bus.record_action(action.model_copy(update={"action_id": str(i)}))
        assert len(bus.actions) == 50
        assert bus.actions[0].sequence == 17
        await bus.override(OverrideRequest(enabled=True, reason="Commander intervention"))
        blocked = await bus.record_action(action.model_copy(update={"action_id": "blocked"}))
        assert blocked.status == "blocked"
        escalation = action.model_copy(
            update={"action_id": "escalation", "action_type": "ESCALATE_HUMAN"}
        )
        assert (await bus.record_action(escalation)).status == "escalated"
        await bus.override(OverrideRequest(enabled=False, reason="All clear"))
        assert (
            await bus.record_action(action.model_copy(update={"action_id": "resume"}))
        ).status == "accepted"
    finally:
        await streamer.close()


async def test_uncertainty_escalates_once_and_recovers(
    settings: Settings, vision: VisionState
) -> None:
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    try:
        for i, occluded in enumerate([True, True, False, True]):
            payload = insert(bus, i)
            state = vision.model_copy(
                update={"frame_id": i, "timestamp": payload.timestamp, "is_occluded": occluded}
            )
            await bus.accept_vision(state)
        assert len(bus.actions) == 2
        assert all(trace.action.action_type == "ESCALATE_HUMAN" for trace in bus.actions)
        assert all(trace.origin == "safety" for trace in bus.actions)
        with pytest.raises(BusConflict):
            await bus.accept_vision(vision.model_copy(update={"frame_id": 999}))
    finally:
        await streamer.close()


async def test_low_confidence_and_dimension_conflicts(
    settings: Settings, vision: VisionState
) -> None:
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    try:
        payload = insert(bus)
        box = BoundingBox(
            coordinates=[10.0, 10.0, 20.0, 20.0],
            centroid=[20.0, 20.0],
            classification="survivor",
            confidence=0.4,
        )
        bad = vision.model_copy(update={"frame_width": 123, "timestamp": payload.timestamp})
        with pytest.raises(BusConflict):
            await bus.accept_vision(bad)
        await bus.accept_vision(
            vision.model_copy(update={"timestamp": payload.timestamp, "bounding_boxes": [box]})
        )
        assert bus.actions[-1].status == "escalated"
    finally:
        await streamer.close()


async def test_broadcast_is_correlated_and_backpressure_bounded(
    settings: Settings, vision: VisionState
) -> None:
    settings = settings.model_copy(update={"vision_wait_ms": 0.0})
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    payload = insert(bus, 0)
    await bus.accept_vision(vision.model_copy(update={"timestamp": payload.timestamp}))
    queue = await bus.subscribe()
    task = asyncio.create_task(bus.broadcast())
    try:
        packet = LiveEnvelope.model_validate_json(await asyncio.wait_for(queue.get(), 1))
        assert packet.frame.frame_id == packet.vision.frame_id == 0
        insert(bus, 1)
        await asyncio.sleep(0.14)
        assert queue.qsize() == 1
        packet = LiveEnvelope.model_validate_json(queue.get_nowait())
        assert packet.frame.frame_id == 1 and packet.vision is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await bus.unsubscribe(queue)
        await streamer.close()


async def test_out_of_order_vision_does_not_rewind_safety(
    settings: Settings, vision: VisionState
) -> None:
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    try:
        for i, occluded in [(2, True), (1, False)]:
            payload = insert(bus, i)
            await bus.accept_vision(
                vision.model_copy(
                    update={"frame_id": i, "timestamp": payload.timestamp, "is_occluded": occluded}
                )
            )
        assert bus._unsafe
        assert len(bus.actions) == 1
    finally:
        await streamer.close()


async def test_encoder_failure_is_visible(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue: asyncio.Queue[CapturedFrame] = asyncio.Queue()
    streamer = VideoStreamer(settings, queue)
    bus = IntegrationBus(settings, streamer)
    packet = streamer._mock(0, datetime.now(UTC))
    queue.put_nowait(replace(packet))

    def broken_encode(*args: object) -> bytes:
        raise ValueError("encoder failed")

    monkeypatch.setattr("src.server.bus.encode_jpeg", broken_encode)
    try:
        with pytest.raises(ValueError, match="encoder failed"):
            await bus.consume_frames(queue)
        assert not bus.frames
    finally:
        await streamer.close()


async def test_late_vision_reports_evidence_age(settings: Settings, vision: VisionState) -> None:
    streamer = VideoStreamer(settings, asyncio.Queue())
    bus = IntegrationBus(settings, streamer)
    try:
        payload = insert(bus)
        stored = bus.frames[0]
        bus.frames[0] = replace(stored, received_monotonic=time.monotonic() - 6)
        await bus.accept_vision(vision.model_copy(update={"timestamp": payload.timestamp}))
        telemetry = await bus.telemetry()
        assert telemetry.telemetry_age_ms >= 6000
        assert any(a.alert_id == "vision-stale" for a in telemetry.active_alerts)
    finally:
        await streamer.close()
