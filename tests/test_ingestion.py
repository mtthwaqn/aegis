import asyncio
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion.video_streamer import CapturedFrame, VideoStreamer
from src.server.config import Settings


async def test_mock_is_paced_and_discards_backlog(settings: Settings) -> None:
    queue: asyncio.Queue[CapturedFrame] = asyncio.Queue(maxsize=1)
    streamer = VideoStreamer(settings, queue)
    task = asyncio.create_task(streamer.run())
    try:
        await asyncio.sleep(0.3)
        frame = await queue.get()
        assert frame.mock_vision.frame_id == frame.frame_id
        assert 2 <= frame.frame_id <= 13
        assert streamer.dropped_frames > 0
        assert queue.qsize() <= 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await streamer.close()
    assert streamer.status == "stopped"


async def test_local_mp4_loops(settings: Settings, tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert writer.isOpened()
    writer.write(np.full((48, 64, 3), 25, np.uint8))
    writer.write(np.full((48, 64, 3), 220, np.uint8))
    writer.release()
    queue: asyncio.Queue[CapturedFrame] = asyncio.Queue(maxsize=2)
    streamer = VideoStreamer(settings.model_copy(update={"video_source": str(source)}), queue)
    task = asyncio.create_task(streamer.run())
    started = time.monotonic()
    try:
        frames = [await asyncio.wait_for(queue.get(), 2) for _ in range(6)]
        assert [frame.frame_id for frame in frames] == list(range(6))
        assert [frame.image.mean() > 100 for frame in frames] == [False, True] * 3
        assert time.monotonic() - started >= 0.13
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await streamer.close()


def test_invalid_source_fails_fast(settings: Settings) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        VideoStreamer(
            settings.model_copy(update={"video_source": "/missing/video.mp4"}), asyncio.Queue()
        )


async def test_connection_drop_retries_and_recovers(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue: asyncio.Queue[CapturedFrame] = asyncio.Queue(maxsize=2)
    streamer = VideoStreamer(
        settings.model_copy(update={"video_source": "rtsp://example.test"}), queue
    )
    calls = 0

    def fake_read():
        nonlocal calls
        calls += 1
        if calls < 3:
            return None
        return np.zeros((48, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(streamer, "_read", fake_read)
    task = asyncio.create_task(streamer.run())
    try:
        frame = await asyncio.wait_for(queue.get(), 2)
        assert frame.frame_id == 0
        assert streamer.decode_failures == 2
        assert streamer.status == "streaming"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await streamer.close()
