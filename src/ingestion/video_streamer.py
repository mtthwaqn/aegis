"""Single-owner decoder with bounded latest-frame delivery and paced simulation."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import boto3
import cv2
import numpy as np
from botocore.config import Config
from numpy.typing import NDArray

from src.server.config import Settings
from src.server.schemas import BoundingBox, Coordinates, VisionState

logger = logging.getLogger(__name__)
SourceStatus = Literal["starting", "streaming", "reconnecting", "stopped", "failed"]


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_id: int
    timestamp: datetime
    captured_monotonic: float
    image: NDArray[np.uint8]
    mock_vision: VisionState | None = None


class VideoStreamer:
    """All OpenCV capture operations execute on one dedicated decoder thread.

    Queue access stays on the asyncio loop. When a consumer is slower than the
    source, discard the oldest queued frame instead of accumulating latency.
    """

    def __init__(self, settings: Settings, queue: asyncio.Queue[CapturedFrame]) -> None:
        self.settings = settings
        self.queue = queue
        self.status: SourceStatus = "starting"
        self.dropped_frames = 0
        self.decode_failures = 0
        self.last_frame_at: float | None = None
        self._capture: cv2.VideoCapture | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-decoder")
        self._closed = False
        self._frame_id = 0
        self._url_expires = 0.0
        self._resolved_url = ""
        source = settings.video_source
        self._is_local = source != "mock" and not urlparse(source).scheme
        if self._is_local and not Path(source).is_file():
            self._executor.shutdown(wait=False)
            raise ValueError("Local video source does not exist")
        if source != "mock" and not self._is_local:
            parsed = urlparse(source)
            if parsed.scheme not in {"rtsp", "rtsps", "http", "https", "kinesis"}:
                self._executor.shutdown(wait=False)
                raise ValueError("Supported sources: local path, RTSP, HTTP HLS, kinesis://name")
            if parsed.scheme == "kinesis" and not parsed.netloc:
                self._executor.shutdown(wait=False)
                raise ValueError("Kinesis source requires a stream name")

    def _resolve_source(self) -> str:
        source = self.settings.video_source
        if not source.startswith("kinesis://"):
            return source
        if self._resolved_url and time.monotonic() < self._url_expires:
            return self._resolved_url
        stream_name = unquote(urlparse(source).netloc)
        config = Config(connect_timeout=3, read_timeout=3, retries={"max_attempts": 2})
        endpoint = self.settings.kinesis_data_endpoint
        if endpoint is None:
            client = boto3.client(
                "kinesisvideo", region_name=self.settings.kinesis_region, config=config
            )
            try:
                endpoint = client.get_data_endpoint(
                    StreamName=stream_name, APIName="GET_HLS_STREAMING_SESSION_URL"
                )["DataEndpoint"]
            finally:
                client.close()
        archived = boto3.client(
            "kinesis-video-archived-media",
            region_name=self.settings.kinesis_region,
            endpoint_url=endpoint,
            config=config,
        )
        try:
            self._resolved_url = archived.get_hls_streaming_session_url(
                StreamName=stream_name,
                PlaybackMode="LIVE",
                HLSFragmentSelector={"FragmentSelectorType": "SERVER_TIMESTAMP"},
                Expires=self.settings.kinesis_session_seconds,
            )["HLSStreamingSessionURL"]
        finally:
            archived.close()
        self._url_expires = time.monotonic() + self.settings.kinesis_session_seconds - 60
        return self._resolved_url

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read(self) -> NDArray[np.uint8] | None:
        if self.settings.video_source.startswith("kinesis://"):
            if time.monotonic() >= self._url_expires:
                self._release()
        if self._capture is None:
            source = self._resolve_source()
            if self._is_local:
                self._capture = cv2.VideoCapture(source)
            else:
                self._capture = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        self.settings.read_timeout_ms,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        self.settings.read_timeout_ms,
                    ],
                )
            if not self._capture.isOpened():
                self._release()
                return None
        ok, frame = self._capture.read()
        if not ok and self._is_local:
            # Seek, then reopen if the container/backend cannot seek accurately.
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok:
                self._release()
                self._capture = cv2.VideoCapture(self.settings.video_source)
                ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            self._release()
            return None
        # Bound bandwidth and allocations for feeds larger than 720p. Contracts
        # always use these decoded dimensions, as supplied to vision consumers.
        height, width = frame.shape[:2]
        ratio = min(1.0, 1280 / width, 720 / height)
        if ratio < 1:
            frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
        return frame

    def _mock(self, frame_id: int, timestamp: datetime) -> CapturedFrame:
        started = time.monotonic()
        width, height = 960, 540
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (47, 35, 17)
        drift = frame_id % 70
        for y in range(-70, height + 70, 70):
            cv2.line(frame, (0, y + drift), (width, y + drift + 25), (66, 57, 28), 2)
        cv2.fillPoly(frame, [np.array([[0, 0], [230, 0], [150, 200], [0, 300]])], (45, 64, 50))
        x = 310.0 + 90 * math.sin(frame_id / 85)
        y = 220.0 + 30 * math.cos(frame_id / 60)
        hazard_x = 610.0 + 20 * math.sin(frame_id / 100)
        occluded = (frame_id // 150) % 3 == 2
        boxes = [
            BoundingBox(
                coordinates=[x, y, 45.0, 68.0],
                centroid=[x + 22.5, y + 34],
                confidence=0.54 if occluded else 0.94,
                classification="survivor",
                geo_coordinates=Coordinates(lat=14.6006, lon=120.9833),
            ),
            BoundingBox(
                coordinates=[hazard_x, 280.0, 165.0, 94.0],
                centroid=[hazard_x + 82.5, 327.0],
                confidence=0.89,
                classification="hazard",
                geo_coordinates=Coordinates(lat=14.6031, lon=120.9883),
            ),
        ]
        # Draw only scene objects; annotations are browser-side and tied to IDs.
        cv2.circle(frame, (round(x + 22), round(y + 24)), 9, (150, 220, 240), -1)
        cv2.line(
            frame,
            (round(x + 22), round(y + 34)),
            (round(x + 22), round(y + 61)),
            (170, 200, 210),
            6,
        )
        cv2.ellipse(frame, (round(hazard_x + 82), 327), (78, 40), 0, 0, 360, (35, 47, 89), -1)
        if occluded:
            overlay = frame.copy()
            overlay[:] = (130, 130, 130)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.putText(
            frame,
            "AEGIS / SYNTHETIC EVALUATION FEED",
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (150, 180, 180),
            1,
        )
        latency = (time.monotonic() - started) * 1000
        state = VisionState(
            frame_id=frame_id,
            timestamp=timestamp,
            latency_ms=latency,
            bounding_boxes=boxes,
            is_occluded=occluded,
            frame_width=width,
            frame_height=height,
            drone_coordinates=Coordinates(
                lat=14.601 + math.sin(frame_id / 200) * 0.002,
                lon=120.985 + math.cos(frame_id / 200) * 0.003,
            ),
        )
        return CapturedFrame(frame_id, timestamp, started, frame, state)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        interval = 1 / self.settings.ingestion_fps
        deadline = loop.time()
        retry_delay = 0.25
        try:
            while not self._closed:
                now = datetime.now(UTC)
                if self.settings.video_source == "mock":
                    packet = await loop.run_in_executor(
                        self._executor, self._mock, self._frame_id, now
                    )
                else:
                    try:
                        pixels = await loop.run_in_executor(self._executor, self._read)
                    except Exception:
                        # Do not log signed URLs, RTSP passwords, or AWS response bodies.
                        logger.warning("Source connection/decoding failed; retrying")
                        pixels = None
                    if pixels is None:
                        self.status = "reconnecting"
                        self.decode_failures += 1
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(5.0, retry_delay * 2)
                        deadline = loop.time()
                        continue
                    captured = time.monotonic()
                    packet = CapturedFrame(self._frame_id, datetime.now(UTC), captured, pixels)
                self.status = "streaming"
                retry_delay = 0.25
                self.last_frame_at = time.monotonic()
                self._frame_id += 1
                if self.queue.full():
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.dropped_frames += 1
                self.queue.put_nowait(packet)
                deadline = max(deadline + interval, loop.time())
                await asyncio.sleep(max(0.001, deadline - loop.time()))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.status = "failed"
            logger.exception("Ingestion task failed")
            raise
        finally:
            if self.status != "failed":
                self.status = "stopped"

    async def close(self) -> None:
        """Call after cancelling run; queued release never races a pending read."""
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._release)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)


def encode_jpeg(frame: NDArray[np.uint8], quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()
