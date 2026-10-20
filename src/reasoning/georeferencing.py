from __future__ import annotations

import math
from dataclasses import dataclass

from src.server.schemas import Coordinates


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fx", "fy", "cx", "cy"):
            finite_number(getattr(self, name), name)
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")
        if not (0 <= self.cx <= self.width and 0 <= self.cy <= self.height):
            raise ValueError("Principal point must lie inside the image")

    @classmethod
    def from_horizontal_fov(
        cls, width: int, height: int, degrees: float
    ) -> CameraIntrinsics:
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        fov = finite_number(degrees, "horizontal_fov_degrees")
        if not 0 < fov < 180:
            raise ValueError("Horizontal field of view must be between 0 and 180 degrees")
        focal = width / (2 * math.tan(math.radians(fov) / 2))
        return cls(width, height, focal, focal, width / 2, height / 2)


@dataclass(frozen=True, slots=True)
class CameraPose:
    coordinates: Coordinates
    height_above_water_m: float
    heading_clockwise_from_north_deg: float
    pitch_down_from_horizon_deg: float
    roll_clockwise_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, Coordinates):
            raise ValueError("coordinates must use the shared Coordinates contract")
        for name in (
            "height_above_water_m", "heading_clockwise_from_north_deg",
            "pitch_down_from_horizon_deg", "roll_clockwise_deg",
        ):
            finite_number(getattr(self, name), name)
        if self.height_above_water_m <= 0:
            raise ValueError("Camera height above water must be positive")
        if not -90 <= self.pitch_down_from_horizon_deg <= 90:
            raise ValueError("Pitch must be between -90 and 90 degrees")


def project_pixel_to_water(
    x: float,
    y: float,
    camera: CameraIntrinsics,
    pose: CameraPose,
    max_horizontal_distance_m: float = 2000.0,
) -> Coordinates | None:
    x, y = finite_number(x, "x"), finite_number(y, "y")
    limit = finite_number(max_horizontal_distance_m, "max_horizontal_distance_m")
    if limit <= 0:
        raise ValueError("Maximum horizontal distance must be positive")
    if not (0 <= x <= camera.width and 0 <= y <= camera.height):
        raise ValueError("Pixel lies outside the camera image")
    yaw = math.radians(pose.heading_clockwise_from_north_deg)
    pitch = math.radians(pose.pitch_down_from_horizon_deg)
    roll = math.radians(pose.roll_clockwise_deg)
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    down = (-math.sin(yaw) * math.sin(pitch),
            -math.cos(yaw) * math.sin(pitch), -math.cos(pitch))
    forward = (math.sin(yaw) * math.cos(pitch),
               math.cos(yaw) * math.cos(pitch), -math.sin(pitch))
    px, py = (x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy
    rx = px * math.cos(roll) - py * math.sin(roll)
    ry = px * math.sin(roll) + py * math.cos(roll)
    east, north, up = (
        forward[i] + rx * right[i] + ry * down[i] for i in range(3)
    )
    if up >= -1e-8:
        return None
    distance_factor = -pose.height_above_water_m / up
    east, north = east * distance_factor, north * distance_factor
    distance = math.hypot(east, north)
    if distance > limit:
        return None
    return _offset_coordinates(pose.coordinates, east, north)


def _offset_coordinates(origin: Coordinates, east: float, north: float) -> Coordinates:
    distance = math.hypot(east, north)
    if distance == 0:
        return origin
    angle = distance / 6371008.8
    bearing = math.atan2(east, north)
    lat, lon = math.radians(origin.lat), math.radians(origin.lon)
    new_lat = math.asin(max(-1.0, min(1.0,
        math.sin(lat) * math.cos(angle)
        + math.cos(lat) * math.sin(angle) * math.cos(bearing)
    )))
    new_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angle) * math.cos(lat),
        math.cos(angle) - math.sin(lat) * math.sin(new_lat),
    )
    return Coordinates(
        lat=math.degrees(new_lat), lon=(math.degrees(new_lon) + 180) % 360 - 180
    )
