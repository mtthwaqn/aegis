from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.ingestion.video_streamer import VideoStreamer
from src.server.config import Settings
from src.server.main import create_app
from src.server.schemas import LiveEnvelope


@pytest.mark.parametrize(
    "classification,confidence,expected_state,geo",
    [
        ("survivor", 0.94, "TARGET_VERIFIED", {"lat": 14.6, "lon": 120.98}),
        ("survivor", 0.4, "LOW_CONFIDENCE", None),
        ("hazard", 0.9, "HAZARD_DETECTED", None),
        (None, None, "NO_TARGET", None),
    ],
)
def test_posted_vision_reaches_websocket_with_all_five_fields(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    classification: str | None,
    confidence: float | None,
    expected_state: str,
    geo: dict[str, float] | None,
) -> None:
    # Keep a real paced/encoded frame path, but supply perception through POST.
    original = VideoStreamer._mock
    monkeypatch.setattr(
        VideoStreamer, "_mock", lambda self, *args: replace(original(self, *args), mock_vision=None)
    )
    settings = settings.model_copy(update={"ingestion_fps": 5.0, "api_token": "vision-test"})
    headers = {"Authorization": "Bearer vision-test"}
    with TestClient(create_app(settings)) as client, client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "auth", "token": "vision-test"})
        for _ in range(60):
            packet = ws.receive_json()
            if packet["frame"] is not None:
                break
        response = client.get("/api/frames/latest", headers=headers)
        frame_id = int(response.headers["X-Frame-ID"])
        boxes = (
            []
            if classification is None
            else [
                {
                    "coordinates": [10.0, 10.0, 20.0, 20.0],
                    "centroid": [20.0, 20.0],
                    "confidence": confidence,
                    "classification": classification,
                    "geo_coordinates": geo,
                }
            ]
        )
        payload = {
            "frame_id": frame_id,
            "timestamp": response.headers["X-Frame-Timestamp"],
            "frame_width": int(response.headers["X-Frame-Width"]),
            "frame_height": int(response.headers["X-Frame-Height"]),
            "latency_ms": 17.0,
            "is_occluded": False,
            "bounding_boxes": boxes,
        }
        assert (
            client.post("/api/telemetry/vision", json=payload, headers=headers).status_code == 202
        )
        for _ in range(60):
            packet = ws.receive_json()
            if packet["reasoning"] and packet["reasoning"]["frame_id"] == frame_id:
                break
        else:
            pytest.fail("Posted VisionState did not reach the live reasoning stream")
        parsed = LiveEnvelope.model_validate(packet)
        reasoning = packet["reasoning"]
        assert parsed.vision.frame_id == parsed.reasoning.frame_id == frame_id
        assert parsed.frame.frame_id >= frame_id
        assert reasoning["state"] == expected_state
        assert reasoning["confidence"]["score"] == confidence
        assert reasoning["hazard"]["detected"] == (classification == "hazard")
        assert reasoning["location"] == geo
        assert (reasoning["target"] is not None) == (classification == "survivor")
        assert reasoning["timestamp"] == packet["vision"]["timestamp"]
        # Prevent future envelope changes from allowing mismatched overlays.
        reasoning["frame_id"] += 1
        with pytest.raises(ValidationError, match="matching video frame"):
            LiveEnvelope.model_validate(packet)


def test_dashboard_script_is_served(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.get("/assets/vision-state.js")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/javascript")
