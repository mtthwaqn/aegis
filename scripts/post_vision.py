"""Continuously acquire gateway frames, run perception, and publish VisionState."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.actions.decide import decide
from src.perception.video_processing import analyze, to_vision_state
from src.reasoning.georeferencing import CameraIntrinsics, CameraPose, project_pixel_to_water
from src.server.config import Settings
from src.server.schemas import Coordinates, VisionState

SYNTHETIC_ORIGIN = Coordinates(lat=14.6010, lon=120.9850)


def synthetic_geo_enabled() -> bool:
    return os.environ.get("AEGIS_SYNTHETIC_GEO", "").strip().lower() in {"1", "true", "yes"}


def with_synthetic_geo(state: VisionState) -> VisionState:
    """Place boxes on a demo camera pose. This is not a surveyed fix for the clip."""
    camera = CameraIntrinsics.from_horizontal_fov(state.frame_width, state.frame_height, 66)
    pose = CameraPose(
        coordinates=SYNTHETIC_ORIGIN,
        height_above_water_m=30.0,
        heading_clockwise_from_north_deg=180.0,
        pitch_down_from_horizon_deg=55.0,
        roll_clockwise_deg=0.0,
    )
    boxes = []
    for box in state.bounding_boxes:
        geo = project_pixel_to_water(box.centroid[0], box.centroid[1], camera, pose)
        boxes.append(box.model_copy(update={"geo_coordinates": geo or SYNTHETIC_ORIGIN}))
    return state.model_copy(update={"bounding_boxes": boxes, "drone_coordinates": SYNTHETIC_ORIGIN})


def submit_latest(base_url: str, api_token: str | None = None) -> tuple[int, dict[str, object]]:
    """Publish one frame, including empty scenes; preserve acquisition identity.

    Geographic enrichment, when enabled, precedes this first immutable POST.
    Without AEGIS_SYNTHETIC_GEO, detections leave geographic coordinates empty.
    """
    token = Settings().api_token if api_token is None else api_token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    latest = Request(f"{base_url.rstrip('/')}/api/frames/latest", headers=headers)
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
    if synthetic_geo_enabled():
        state = with_synthetic_geo(state)
    request = Request(
        f"{base_url.rstrip('/')}/api/telemetry/vision",
        data=state.model_dump_json().encode(),
        headers=headers | {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode())
            status = response.status
    except HTTPError as error:
        # Read the error before closing its underlying connection.
        with error:
            body = json.loads(error.read().decode())
            status = error.code
    if status == 202 and synthetic_geo_enabled():
        _post_decision(base_url, headers, state)
    return status, {
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


def _post_decision(base_url: str, headers: dict[str, str], state: VisionState) -> None:
    decision = decide(state, threshold=Settings().low_confidence_threshold)
    if decision.action is None or decision.outcome not in {"dispatch_payload", "reroute_ground_rescue"}:
        return
    action = decision.action.model_copy(
        update={
            "action_id": f"clip-frame-{state.frame_id}",
            "reasoning_trace": (
                f"{decision.reason} Synthetic camera pose for this clip, not a surveyed GPS fix."
            ),
        }
    )
    request = Request(
        f"{base_url.rstrip('/')}/api/agent/action",
        data=action.model_dump_json().encode(),
        headers=headers | {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5):
            return
    except HTTPError:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--attempts", type=int, help="Stop after this many polling attempts")
    runs.add_argument("--once", action="store_true", help="Process one acquired frame and exit")
    parser.add_argument(
        "--interval", type=float, default=0.1, help="Pause between frames in seconds"
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 0.01:
        parser.error("--interval must be finite and at least 0.01 seconds")
    if args.attempts is not None and args.attempts < 1:
        parser.error("--attempts must be positive")
    limit = 1 if args.once else args.attempts
    token = Settings().api_token
    attempts = 0
    accepted = 0
    try:
        while limit is None or attempts < limit:
            attempts += 1
            try:
                status, summary = submit_latest(args.base_url, token)
                print(json.dumps(summary), flush=True)
                if status == 202:
                    accepted += 1
                elif status != 409:
                    return 1
            except HTTPError as error:
                with error:
                    print(f"Frame acquisition rejected (HTTP {error.code})", file=sys.stderr)
                    if error.code in {401, 403} or args.once:
                        return 1
            except (URLError, TimeoutError, OSError, ValueError, RuntimeError, cv2.error):
                print("Vision acquisition/processing failed; retrying", file=sys.stderr)
                if args.once:
                    return 1
            if limit is None or attempts < limit:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
