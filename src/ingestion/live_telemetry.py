"""Read a Skydio demo telemetry websocket and place boxes on its GPS footprint."""

from __future__ import annotations

import json
import os
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from src.server.schemas import Coordinates, VisionState

STALE_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class LiveFix:
    drone: Coordinates
    received_monotonic: float
    center: Coordinates | None = None
    top_left: Coordinates | None = None
    top_right: Coordinates | None = None
    bottom_left: Coordinates | None = None
    bottom_right: Coordinates | None = None


def _pair(node: object, lat_key: str, lon_key: str) -> Coordinates | None:
    if not isinstance(node, dict):
        return None
    try:
        lat = float(node[lat_key])
        lon = float(node[lon_key])
    except (KeyError, TypeError, ValueError):
        return None
    if lat != lat or lon != lon or abs(lat) > 90 or abs(lon) > 180:
        return None
    if lat == 0 and lon == 0:
        return None
    return Coordinates(lat=lat, lon=lon)


def parse_status(payload: object, *, received_monotonic: float) -> LiveFix | None:
    """Keep the drone fix and the camera footprint. Ignore every other field."""
    if not isinstance(payload, dict) or payload.get("type") not in {None, "status"}:
        return None
    drone = _pair(payload, "lat", "lon")
    if drone is None:
        return None
    footprint = payload.get("camera_projections")
    if not isinstance(footprint, dict):
        footprint = {}
    return LiveFix(
        drone=drone,
        received_monotonic=received_monotonic,
        center=_pair(footprint.get("center"), "latitude", "longitude"),
        top_left=_pair(footprint.get("top_left"), "latitude", "longitude"),
        top_right=_pair(footprint.get("top_right"), "latitude", "longitude"),
        bottom_left=_pair(footprint.get("bottom_left"), "latitude", "longitude"),
        bottom_right=_pair(footprint.get("bottom_right"), "latitude", "longitude"),
    )


def _lerp(origin: Coordinates, dest: Coordinates, amount: float) -> Coordinates:
    return Coordinates(
        lat=origin.lat + (dest.lat - origin.lat) * amount,
        lon=origin.lon + (dest.lon - origin.lon) * amount,
    )


def footprint_at(x: float, y: float, width: int, height: int, fix: LiveFix) -> Coordinates:
    """Map a pixel onto the camera footprint. Fall back to the drone fix."""
    corners = (fix.top_left, fix.top_right, fix.bottom_left, fix.bottom_right)
    if not all(corners) or width <= 0 or height <= 0:
        return fix.center or fix.drone
    across = min(1.0, max(0.0, x / width))
    down = min(1.0, max(0.0, y / height))
    top = _lerp(fix.top_left, fix.top_right, across)  # type: ignore[arg-type]
    bottom = _lerp(fix.bottom_left, fix.bottom_right, across)  # type: ignore[arg-type]
    return _lerp(top, bottom, down)


def apply_live_fix(state: VisionState, fix: LiveFix) -> VisionState:
    """Keep the aircraft fix. A box gets a coordinate only from a real camera footprint.

    The demo often sends a drone position without the four ground corners. Copying
    that one point onto every box stacks houses and clouds on the aircraft pin.
    """
    corners = (fix.top_left, fix.top_right, fix.bottom_left, fix.bottom_right)
    boxes = []
    for box in state.bounding_boxes:
        geo = (
            footprint_at(box.centroid[0], box.centroid[1], state.frame_width, state.frame_height, fix)
            if all(corners)
            else None
        )
        boxes.append(box.model_copy(update={"geo_coordinates": geo}))
    return state.model_copy(update={"bounding_boxes": boxes, "drone_coordinates": fix.drone})


class TelemetryFeed:
    """Background reader. The latest fix expires after five seconds."""

    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "wss" or not parsed.hostname:
            raise ValueError("Live telemetry requires a wss:// URL")
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._path = parsed.path or "/"
        if parsed.query:
            self._path = f"{self._path}?{parsed.query}"
        self._lock = threading.Lock()
        self._fix: LiveFix | None = None
        self._thread = threading.Thread(target=self._run, name="aegis-telemetry", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> LiveFix | None:
        with self._lock:
            fix = self._fix
        if fix is None or time.monotonic() - fix.received_monotonic > STALE_SECONDS:
            return None
        return fix

    def _store(self, payload: object) -> None:
        fix = parse_status(payload, received_monotonic=time.monotonic())
        if fix is None:
            return
        with self._lock:
            self._fix = fix

    def _run(self) -> None:
        while True:
            try:
                self._session()
            except Exception:
                time.sleep(1)

    def _session(self) -> None:
        context = ssl.create_default_context()
        raw = socket.create_connection((self._host, self._port), timeout=8)
        try:
            connection = context.wrap_socket(raw, server_hostname=self._host)
            key = os.urandom(16)
            import base64

            handshake = (
                f"GET {self._path} HTTP/1.1\r\n"
                f"Host: {self._host}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {base64.b64encode(key).decode()}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            connection.sendall(handshake.encode())
            buffer = b""
            while b"\r\n\r\n" not in buffer:
                chunk = connection.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            head, buffer = buffer.split(b"\r\n\r\n", 1)
            if b" 101 " not in head.split(b"\r\n", 1)[0]:
                return
            connection.settimeout(30)
            pending = ""
            while True:
                try:
                    text, buffer = _next_text(connection, buffer)
                except socket.timeout:
                    continue
                if text is None:
                    return
                pending += text
                try:
                    payload = json.loads(pending)
                except json.JSONDecodeError:
                    if len(pending) > 1_000_000:
                        pending = ""
                    continue
                pending = ""
                self._store(payload)
        finally:
            raw.close()


def _next_text(connection: socket.socket, buffer: bytes) -> tuple[str | None, bytes]:
    while True:
        while len(buffer) < 2:
            chunk = connection.recv(4096)
            if not chunk:
                return None, buffer
            buffer += chunk
        first, second = buffer[0], buffer[1]
        offset = 2
        length = second & 0x7F
        if length == 126:
            filled = _fill(connection, buffer, 4)
            if filled is None:
                return None, buffer
            buffer = filled
            length = int.from_bytes(buffer[2:4], "big")
            offset = 4
        elif length == 127:
            filled = _fill(connection, buffer, 10)
            if filled is None:
                return None, buffer
            buffer = filled
            length = int.from_bytes(buffer[2:10], "big")
            offset = 10
        while len(buffer) < offset + length:
            chunk = connection.recv(4096)
            if not chunk:
                return None, buffer
            buffer += chunk
        payload = buffer[offset : offset + length]
        buffer = buffer[offset + length :]
        opcode = first & 0x0F
        if opcode == 0x8:
            return None, buffer
        if opcode == 0x9:
            _send_pong(connection, payload)
            continue
        if opcode == 0x1:
            return payload.decode("utf-8", "replace"), buffer


def _fill(connection: socket.socket, buffer: bytes, size: int) -> bytes | None:
    while len(buffer) < size:
        chunk = connection.recv(4096)
        if not chunk:
            return None
        buffer += chunk
    return buffer


def _send_pong(connection: socket.socket, payload: bytes) -> None:
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    header = bytes([0x8A, 0x80 | len(payload)]) if len(payload) < 126 else b""
    if not header:
        return
    connection.sendall(header + mask + masked)
