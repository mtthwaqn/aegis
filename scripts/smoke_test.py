"""Read-only live gateway verification; run after starting Uvicorn."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from websockets.asyncio.client import connect

from src.server.schemas import LiveEnvelope


async def inspect(base_url: str, duration: float) -> None:
    parsed = urlparse(urljoin(base_url.rstrip("/") + "/", "ws/live"))
    ws_url = urlunparse(parsed._replace(scheme="wss" if parsed.scheme == "https" else "ws"))
    token = os.environ.get("AEGIS_API_TOKEN", "")
    received: list[float] = []
    frame_ids: set[int] = set()
    async with connect(ws_url, max_size=8 * 1024 * 1024, open_timeout=5) as websocket:
        if token:
            await websocket.send(json.dumps({"type": "auth", "token": token}))
        started = time.monotonic()
        while time.monotonic() - started < duration:
            envelope = LiveEnvelope.model_validate_json(
                await asyncio.wait_for(websocket.recv(), timeout=5)
            )
            received.append(time.monotonic())
            if envelope.frame is not None:
                frame_ids.add(envelope.frame.frame_id)
                if envelope.vision is not None:
                    if envelope.vision.frame_id != envelope.frame.frame_id:
                        raise RuntimeError("Frame / vision correlation violated")
                    if envelope.reasoning is None:
                        raise RuntimeError("Vision frame is missing its reasoning state")
        if len(received) < 2 or not frame_ids:
            raise RuntimeError("Insufficient live frames")
        measured_hz = (len(received) - 1) / (received[-1] - received[0])
        print(
            json.dumps(
                {
                    "delivered_hz": round(measured_hz, 1),
                    "unique_frames": len(frame_ids),
                    "acquired_fps": round(envelope.telemetry.active_fps, 1),
                    "vision_fps": round(envelope.telemetry.vision_fps, 1),
                    "audit_entries": len(envelope.telemetry.action_log),
                },
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()
    if not 1 <= args.duration <= 60:
        parser.error("duration must be between 1 and 60 seconds")
    health = Request(urljoin(args.base_url.rstrip("/") + "/", "healthz"))
    with urlopen(health, timeout=5) as response:
        print(response.read().decode())
    asyncio.run(inspect(args.base_url, args.duration))


if __name__ == "__main__":
    main()
