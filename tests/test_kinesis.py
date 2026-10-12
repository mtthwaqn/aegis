"""AWS adapter contract tests use stubs; they do not certify a live AWS stream."""

import asyncio
import time
from typing import Any

import pytest

from src.ingestion.video_streamer import VideoStreamer
from src.server.config import Settings


async def test_kinesis_discovers_endpoint_and_renews_session(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def get_data_endpoint(self, **kwargs: Any) -> dict[str, str]:
            calls.append(("endpoint", kwargs))
            return {"DataEndpoint": "https://example.test"}

        def get_hls_streaming_session_url(self, **kwargs: Any) -> dict[str, str]:
            calls.append(("hls", kwargs))
            return {"HLSStreamingSessionURL": "https://example.test/signed.m3u8"}

        def close(self) -> None:
            calls.append(("close", {}))

    def client(service: str, **kwargs: Any) -> Client:
        calls.append((service, kwargs))
        return Client()

    monkeypatch.setattr("src.ingestion.video_streamer.boto3.client", client)
    streamer = VideoStreamer(
        settings.model_copy(update={"video_source": "kinesis://rescue"}), asyncio.Queue()
    )
    try:
        assert streamer._resolve_source() == "https://example.test/signed.m3u8"
        assert streamer._resolve_source() == "https://example.test/signed.m3u8"
        assert len([call for call in calls if call[0] == "hls"]) == 1
        endpoint = next(value for method, value in calls if method == "endpoint")
        assert endpoint == {"StreamName": "rescue", "APIName": "GET_HLS_STREAMING_SESSION_URL"}
        hls = next(value for method, value in calls if method == "hls")
        assert hls["PlaybackMode"] == "LIVE"
        assert hls["Expires"] == settings.kinesis_session_seconds
        streamer._url_expires = time.monotonic() - 1
        streamer._resolve_source()
        assert len([call for call in calls if call[0] == "hls"]) == 2
    finally:
        await streamer.close()
