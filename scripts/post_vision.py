"""Fetch the latest frame, run perception, and submit the vision state."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.perception.video_processing import analyze, to_vision_state


def submit_latest(base_url: str) -> tuple[int, dict[str, object]]:
    latest = Request(f"{base_url.rstrip('/')}/api/frames/latest")
    with urlopen(latest, timeout=5) as response:
        frame_id = int(response.headers["X-Frame-ID"])
        timestamp = datetime.fromisoformat(response.headers["X-Frame-Timestamp"])
        width = int(response.headers["X-Frame-Width"])
        height = int(response.headers["X-Frame-Height"])
        encoded = response.read()
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or (frame.shape[1], frame.shape[0]) != (width, height):
        raise RuntimeError("Decoded frame does not match the acquisition headers")
    result = analyze(frame)
    state = to_vision_state(result, frame_id, timestamp)
    payload = state.model_dump_json().encode()
    request = Request(
        f"{base_url.rstrip('/')}/api/telemetry/vision",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode())
            status = response.status
    except HTTPError as error:
        body = json.loads(error.read().decode())
        status = error.code
    summary = {
        "http_status": status,
        "response": body,
        "frame_id": frame_id,
        "detections": [
            {
                "label": detection.label,
                "classification": detection.classification,
                "confidence": round(detection.confidence, 2),
                "coordinates": [round(value, 1) for value in detection.coordinates],
            }
            for detection in result.detections
        ],
    }
    return status, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--attempts", type=int, default=80)
    args = parser.parse_args()
    last: dict[str, object] = {}
    for _ in range(args.attempts):
        status, last = submit_latest(args.base_url)
        detections = last["detections"]
        labels = (
            {str(item["label"]) for item in detections} if isinstance(detections, list) else set()
        )
        if status == 202 and "PERSON" in labels:
            print(json.dumps(last, indent=2))
            return
        if status not in {202, 409}:
            print(json.dumps(last, indent=2))
            raise SystemExit(1)
        time.sleep(0.75)
    print(json.dumps(last, indent=2))
    raise SystemExit("No accepted person detection in the sampled frames")


if __name__ == "__main__":
    main()
