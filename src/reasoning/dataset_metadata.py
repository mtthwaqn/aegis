from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.server.schemas import Coordinates, VisionState

from .georeferencing import (
    CameraIntrinsics,
    CameraPose,
    finite_number,
    project_pixel_to_water,
)


def load_image_record(path: str | Path, image_id: int) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = document.get("images") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise ValueError("Expected a COCO JSON object containing an images array")
    matches = [
        record for record in records if isinstance(record, dict) and record.get("id") == image_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one image record for image_id={image_id}")
    return matches[0]


def enrich_from_seadronessee(
    vision: VisionState,
    image_record: dict[str, Any],
    horizontal_fov_degrees: float | None = None,
) -> tuple[VisionState, dict[str, Any]]:
    meta = image_record.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("Image meta must be a JSON object")
    report: dict[str, Any] = {
        "source": "SeaDronesSee",
        "image_id": image_record.get("id"),
        "file_name": image_record.get("file_name"),
        "source_frame": image_record.get("source"),
        "status": "unavailable",
        "reason": None,
        "boxes": [],
        "assumptions": [],
    }
    missing = [key for key in ("gps_latitude", "gps_longitude") if meta.get(key) is None]
    if missing:
        report["reason"] = "Missing metadata: " + ", ".join(missing)
        return vision, report
    drone = Coordinates(
        lat=_signed_coordinate(meta, "gps_latitude", "N", "S"),
        lon=_signed_coordinate(meta, "gps_longitude", "E", "W"),
    )
    if vision.drone_coordinates is not None and vision.drone_coordinates != drone:
        raise ValueError("Metadata GPS conflicts with existing frame drone_coordinates")
    payload = vision.model_dump(mode="python")
    payload["drone_coordinates"] = drone
    report["status"] = "drone_only"
    required = ("altitude", "gimbal_pitch", "gimbal_heading")
    missing = [key for key in required if meta.get(key) is None]
    if horizontal_fov_degrees is None or missing:
        report["reason"] = (
            "Target projection requires an explicit horizontal field of view"
            if horizontal_fov_degrees is None
            else "Missing metadata: " + ", ".join(missing)
        )
        return VisionState.model_validate(payload), report
    width, height = image_record.get("width"), image_record.get("height")
    camera = CameraIntrinsics.from_horizontal_fov(width, height, horizontal_fov_degrees)
    if not math.isclose(width / height, vision.frame_width / vision.frame_height, rel_tol=0.005):
        raise ValueError("Source and detection aspect ratios differ; cropping needs a transform")
    pose = CameraPose(
        coordinates=drone,
        height_above_water_m=finite_number(meta["altitude"], "altitude"),
        heading_clockwise_from_north_deg=finite_number(meta["gimbal_heading"], "gimbal_heading"),
        pitch_down_from_horizon_deg=finite_number(meta["gimbal_pitch"], "gimbal_pitch"),
        roll_clockwise_deg=finite_number(meta.get("gimbal_roll", 0.0), "gimbal_roll"),
    )
    report["assumptions"] = [
        "Flat water plane; metadata altitude is treated as height above that plane",
        "Pinhole camera, square pixels, centered principal point; lens distortion uncorrected",
        "Bounding-box centroid lies on the water surface",
        "Source image is resized without cropping or letterboxing",
        "Short-range spherical Earth conversion; horizontal range limited to 2000 m",
    ]
    if "gimbal_roll" not in meta:
        report["assumptions"].append("Camera roll assumed zero because metadata omits it")
    report["horizontal_fov_degrees"] = float(horizontal_fov_degrees)
    report["status"] = "estimated"
    successful = 0
    for index, box in enumerate(vision.bounding_boxes):
        if box.geo_coordinates is not None:
            report["boxes"].append({"index": index, "status": "provided"})
            continue
        cx, cy = box.centroid
        location = project_pixel_to_water(
            cx * width / vision.frame_width,
            cy * height / vision.frame_height,
            camera,
            pose,
        )
        payload["bounding_boxes"][index]["geo_coordinates"] = location
        successful += location is not None
        report["boxes"].append(
            {
                "index": index,
                "status": "estimated" if location is not None else "unavailable",
                "reason": None
                if location is not None
                else "Ray misses water or exceeds range limit",
            }
        )
    if not successful:
        report["status"] = (
            "provided"
            if any(box.geo_coordinates is not None for box in vision.bounding_boxes)
            else "drone_only"
        )
        report["reason"] = "No new target coordinate was projected"
    return VisionState.model_validate(payload), report


def _signed_coordinate(meta: dict[str, Any], key: str, positive: str, negative: str) -> float:
    number = finite_number(meta[key], key)
    reference = meta.get(key + "_ref")
    if reference is None:
        return number
    if reference not in (positive, negative):
        raise ValueError(f"Invalid hemisphere reference for {key}")
    if reference == positive and number < 0:
        raise ValueError(f"Conflicting sign and hemisphere reference for {key}")
    return -abs(number) if reference == negative else abs(number)
