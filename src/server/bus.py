"""Bounded, loop-owned integration state; no shared decoder-thread mutation."""

from __future__ import annotations

import asyncio
import base64
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime

from src.ingestion.video_streamer import CapturedFrame, VideoStreamer, encode_jpeg
from src.server.config import Settings
from src.server.schemas import (
    AgentActionPayload,
    Alert,
    ControlState,
    Coordinates,
    DecisionTrace,
    LiveEnvelope,
    OverrideRequest,
    SystemTelemetry,
    VideoFramePayload,
    VisionState,
)


class BusConflict(ValueError):
    """An event cannot be correlated or an idempotency key was reused."""


@dataclass(frozen=True, slots=True)
class StoredFrame:
    payload: VideoFramePayload
    jpeg: bytes
    received_monotonic: float


class IntegrationBus:
    def __init__(self, settings: Settings, streamer: VideoStreamer) -> None:
        self.settings = settings
        self.streamer = streamer
        self.lock = asyncio.Lock()
        self.frames: OrderedDict[int, StoredFrame] = OrderedDict()
        self.vision: OrderedDict[int, VisionState] = OrderedDict()
        self.actions: deque[DecisionTrace] = deque(maxlen=50)
        self._seen_actions: OrderedDict[str, DecisionTrace] = OrderedDict()
        self.subscribers: set[asyncio.Queue[str]] = set()
        self._frame_times: deque[float] = deque(maxlen=10000)
        self._vision_times: deque[float] = deque(maxlen=10000)
        self._latencies: deque[tuple[float, float]] = deque(maxlen=10000)
        self._started = time.monotonic()
        self._last_vision_at: float | None = None
        self._decision_sequence = 0
        self._broadcast_sequence = 0
        self._unsafe = False
        self._last_mock_action = 0.0
        self._agent_last_seen: float | None = None
        self.control = ControlState(
            override_enabled=False, reason="Autonomy enabled", updated_at=datetime.now(UTC)
        )

    def _record_action(self, action: AgentActionPayload, origin: str) -> DecisionTrace:
        existing = self._seen_actions.get(action.action_id)
        if existing is not None:
            if existing.action != action:
                raise BusConflict("action_id already belongs to a different payload")
            return existing
        self._decision_sequence += 1
        status = "accepted"
        if action.action_type == "ESCALATE_HUMAN":
            status = "escalated"
        elif self.control.override_enabled:
            status = "blocked"
        stages = ["thought", "spatial_evaluation"]
        # This bus records events; it never asserts that hardware completed a mission.
        stages.append(
            "human_escalation"
            if status == "escalated"
            else "action_blocked"
            if status == "blocked"
            else "action_dispatched"
        )
        trace = DecisionTrace(
            sequence=self._decision_sequence,
            action=action,
            received_at=datetime.now(UTC),
            status=status,
            origin=origin,
            stages=stages,
        )
        self.actions.append(trace)
        self._seen_actions[action.action_id] = trace
        if len(self._seen_actions) > 1000:
            self._seen_actions.popitem(last=False)
        return trace

    async def record_action(self, action: AgentActionPayload) -> DecisionTrace:
        async with self.lock:
            self._agent_last_seen = time.monotonic()
            return self._record_action(action, "agent")

    async def override(self, request: OverrideRequest) -> ControlState:
        async with self.lock:
            self.control = ControlState(
                override_enabled=request.enabled,
                reason=request.reason,
                updated_at=datetime.now(UTC),
            )
            action = AgentActionPayload(
                action_type="ESCALATE_HUMAN",
                target_coordinates=Coordinates(lat=0.0, lon=0.0),
                payload_type="commander_control",
                reasoning_trace=request.reason,
                timestamp=datetime.now(UTC),
            )
            self._record_action(action, "safety")
            return self.control

    def _put_vision(self, state: VisionState) -> None:
        stored = self.frames.get(state.frame_id)
        if stored is None:
            raise BusConflict("Unknown or expired frame_id; read /api/frames/latest first")
        if (state.frame_width, state.frame_height) != (stored.payload.width, stored.payload.height):
            raise BusConflict("Vision dimensions do not match the acquired frame")
        if abs((state.timestamp - stored.payload.timestamp).total_seconds()) > 1:
            raise BusConflict("Vision timestamp must identify the acquired frame timestamp")
        existing = self.vision.get(state.frame_id)
        if existing is not None:
            if existing != state:
                raise BusConflict("Vision frame_id already has a different immutable state")
            return
        now = time.monotonic()
        self.vision[state.frame_id] = state
        if len(self.vision) > self.settings.frame_cache_size:
            self.vision.popitem(last=False)
        self._vision_times.append(now)
        self._latencies.append((now, state.latency_ms))
        # Out-of-order states are cached for correlation but never rewind safety state.
        if state.frame_id != max(self.vision):
            return
        # Freshness describes the evidence, not the time a delayed POST arrived.
        self._last_vision_at = stored.received_monotonic
        uncertain = [
            box
            for box in state.bounding_boxes
            if box.confidence < self.settings.low_confidence_threshold
        ]
        unsafe = state.is_occluded or bool(uncertain)
        if unsafe and not self._unsafe:
            geo = next((b.geo_coordinates for b in uncertain if b.geo_coordinates), None)
            geo = geo or state.drone_coordinates or Coordinates(lat=0.0, lon=0.0)
            self._record_action(
                AgentActionPayload(
                    action_id=f"safety-frame-{state.frame_id}",
                    action_type="ESCALATE_HUMAN",
                    target_coordinates=geo,
                    payload_type="operator_review",
                    reasoning_trace="Perception degraded or confidence below threshold; "
                    "human review required. Coordinates may be drone position; "
                    "0,0 means unavailable and must not be used for dispatch.",
                    source_frame_id=state.frame_id,
                    timestamp=datetime.now(UTC),
                ),
                "safety",
            )
        self._unsafe = unsafe

    async def accept_vision(self, state: VisionState) -> None:
        async with self.lock:
            self._put_vision(state)

    def _mock_branch(self, state: VisionState) -> None:
        now = time.monotonic()
        if not self.settings.mock_actions or now - self._last_mock_action < 3:
            return
        self._last_mock_action = now
        if state.is_occluded or any(
            b.confidence < self.settings.low_confidence_threshold for b in state.bounding_boxes
        ):
            action_type = "ESCALATE_HUMAN"
            payload = "operator_review"
            reason = "Synthetic perception occluded → confidence degraded → escalate to commander"
            box = state.bounding_boxes[0]
        elif (state.frame_id // 90) % 2:
            action_type = "REROUTE_CONVOY"
            payload = "hazard_avoidance"
            reason = "Synthetic hazard detected → convoy path intersects hazard → divert convoy"
            box = next(b for b in state.bounding_boxes if b.classification == "hazard")
        else:
            action_type = "DISPATCH_SUPPLY_DRONE"
            payload = "flotation_and_medical"
            reason = "Synthetic survivor confirmed → accessible drop zone → dispatch supply drone"
            box = next(b for b in state.bounding_boxes if b.classification == "survivor")
        self._record_action(
            AgentActionPayload(
                action_type=action_type,
                target_coordinates=box.geo_coordinates,
                payload_type=payload,
                reasoning_trace=reason,
                source_frame_id=state.frame_id,
                timestamp=datetime.now(UTC),
                route_coordinates=[
                    Coordinates(lat=14.597, lon=120.983),
                    Coordinates(lat=14.602, lon=120.981),
                    Coordinates(lat=14.607, lon=120.986),
                ]
                if action_type == "REROUTE_CONVOY"
                else [],
            ),
            "mock",
        )

    async def consume_frames(self, queue: asyncio.Queue[CapturedFrame]) -> None:
        while True:
            frame = await queue.get()
            try:
                # Encoding is off-loop and runs once per frame, not once per viewer.
                jpeg = await asyncio.to_thread(encode_jpeg, frame.image, self.settings.jpeg_quality)
                height, width = frame.image.shape[:2]
                payload = VideoFramePayload(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    width=width,
                    height=height,
                    jpeg_base64=base64.b64encode(jpeg).decode("ascii"),
                )
                async with self.lock:
                    self.frames[frame.frame_id] = StoredFrame(payload, jpeg, time.monotonic())
                    self._frame_times.append(frame.captured_monotonic)
                    if len(self.frames) > self.settings.frame_cache_size:
                        expired, _ = self.frames.popitem(last=False)
                        self.vision.pop(expired, None)
                    if frame.mock_vision is not None:
                        self._put_vision(frame.mock_vision)
                        self._mock_branch(frame.mock_vision)
            finally:
                queue.task_done()

    def _telemetry(self, now: float) -> SystemTelemetry:
        cutoff = now - 30
        for series in (self._frame_times, self._vision_times):
            while series and series[0] < cutoff:
                series.popleft()
        while self._latencies and self._latencies[0][0] < cutoff:
            self._latencies.popleft()
        window = max(0.001, min(30, now - self._started))
        active_fps = len(self._frame_times) / window
        latencies = sorted(value for _, value in self._latencies)
        p95 = latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else 0.0
        alerts: list[Alert] = []
        if self.control.override_enabled:
            alerts.append(
                Alert(alert_id="override", severity="critical", message=self.control.reason)
            )
        if (
            self.streamer.status != "streaming"
            or self.streamer.last_frame_at is None
            or (now - self.streamer.last_frame_at > self.settings.source_stale_seconds)
        ):
            alerts.append(
                Alert(alert_id="source", severity="critical", message="Video source unavailable")
            )
        latest = self.vision[max(self.vision)] if self.vision else None
        if latest and self._unsafe:
            alerts.append(
                Alert(
                    alert_id="vision-review",
                    severity="warning",
                    message="Low confidence / occlusion: human review required",
                    frame_id=latest.frame_id,
                )
            )
        age = (now - self._last_vision_at) * 1000 if self._last_vision_at is not None else None
        if age is None or age > self.settings.source_stale_seconds * 1000:
            alerts.append(
                Alert(
                    alert_id="vision-stale",
                    severity="warning",
                    message="Vision telemetry unavailable or stale",
                )
            )
        cost = None
        if self.settings.hourly_compute_cost_usd is not None and active_fps > 0:
            cost = self.settings.hourly_compute_cost_usd * 100000 / (active_fps * 3600)
        fleet = "simulation" if self.settings.video_source == "mock" else "awaiting_agent"
        if self._agent_last_seen is not None and now - self._agent_last_seen < 30:
            fleet = "agent_connected"
        if self.control.override_enabled:
            fleet = "human_control"
        return SystemTelemetry(
            timestamp=datetime.now(UTC),
            active_fps=active_fps,
            vision_fps=len(self._vision_times) / window,
            p95_latency_ms=p95,
            cloud_compute_instance=self.settings.compute_instance,
            deployment_verified=self.settings.deployment_verified,
            estimated_cost_usd_per_100k_frames=cost,
            fleet_status=fleet,
            active_alerts=alerts,
            action_log=list(self.actions),
            control=self.control,
            source_status=self.streamer.status,
            dropped_frames=self.streamer.dropped_frames,
            connected_viewers=len(self.subscribers),
            metrics_window_seconds=window,
            telemetry_age_ms=age,
        )

    async def telemetry(self) -> SystemTelemetry:
        async with self.lock:
            return self._telemetry(time.monotonic())

    async def latest_frame(self) -> StoredFrame | None:
        async with self.lock:
            return next(reversed(self.frames.values()), None)

    async def subscribe(self) -> asyncio.Queue[str]:
        async with self.lock:
            if len(self.subscribers) >= 64:
                raise BusConflict("Viewer capacity reached")
            queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
            self.subscribers.add(queue)
            return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        async with self.lock:
            self.subscribers.discard(queue)

    async def broadcast(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            async with self.lock:
                now = time.monotonic()
                selected = next(
                    (
                        frame
                        for frame in reversed(self.frames.values())
                        if frame.payload.frame_id in self.vision
                        or (now - frame.received_monotonic) * 1000 >= self.settings.vision_wait_ms
                    ),
                    None,
                )
                self._broadcast_sequence += 1
                envelope = LiveEnvelope(
                    sequence=self._broadcast_sequence,
                    frame=selected.payload if selected else None,
                    vision=self.vision.get(selected.payload.frame_id) if selected else None,
                    telemetry=self._telemetry(now),
                )
                if self.subscribers:
                    serialized = envelope.model_dump_json()
                    for queue in self.subscribers:
                        if queue.full():
                            queue.get_nowait()
                        queue.put_nowait(serialized)
            deadline = max(deadline + 1 / self.settings.broadcast_hz, loop.time())
            await asyncio.sleep(max(0.001, deadline - loop.time()))
