import io
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import pytest

from scripts import post_vision


def test_publisher_preserves_identity_and_authenticates_both_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 100, dtype=np.uint8))
    assert ok
    timestamp = datetime.now(UTC).isoformat()
    acquired = MagicMock()
    acquired.__enter__.return_value = acquired
    acquired.headers = {
        "X-Frame-ID": "42",
        "X-Frame-Timestamp": timestamp,
        "X-Frame-Width": "64",
        "X-Frame-Height": "48",
    }
    acquired.read.return_value = encoded.tobytes()
    accepted = MagicMock()
    accepted.__enter__.return_value = accepted
    accepted.status = 202
    accepted.read.return_value = b'{"status":"accepted","frame_id":42}'
    opener = MagicMock(side_effect=[acquired, accepted])
    monkeypatch.setattr(post_vision, "urlopen", opener)
    status, summary = post_vision.submit_latest("http://gateway.test", "test-token")
    assert status == 202
    assert summary["detections"] == []  # Empty scenes still reach the dashboard.
    get, post = [call.args[0] for call in opener.call_args_list]
    assert (
        get.get_header("Authorization") == post.get_header("Authorization") == "Bearer test-token"
    )
    payload = json.loads(post.data)
    assert payload["frame_id"] == 42
    assert datetime.fromisoformat(payload["timestamp"]) == datetime.fromisoformat(timestamp)
    assert payload["bounding_boxes"] == []


def test_publisher_keeps_running_after_a_person_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    publish = MagicMock(return_value=(202, {"detections": [{"label": "PERSON"}]}))
    monkeypatch.setattr(post_vision, "submit_latest", publish)
    monkeypatch.setattr(post_vision.time, "sleep", lambda _: None)
    assert post_vision.main(["--attempts", "3"]) == 0
    assert publish.call_count == 3


def test_publisher_recovers_from_connection_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    publish = MagicMock(side_effect=[URLError("dropped"), (202, {"detections": []})])
    monkeypatch.setattr(post_vision, "submit_latest", publish)
    monkeypatch.setattr(post_vision.time, "sleep", lambda _: None)
    assert post_vision.main(["--attempts", "2"]) == 0


def test_publisher_exits_on_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    error = HTTPError("http://gateway.test", 401, "Unauthorized", {}, io.BytesIO())
    monkeypatch.setattr(post_vision, "submit_latest", MagicMock(side_effect=error))
    assert post_vision.main(["--attempts", "3"]) == 1
