from datetime import UTC, datetime

from src.ingestion.live_telemetry import LiveFix, apply_live_fix, footprint_at, parse_status
from src.server.schemas import BoundingBox, Coordinates, VisionState


def test_status_keeps_the_drone_fix_and_rejects_the_origin() -> None:
    fix = parse_status(
        {"type": "status", "lat": 37.522, "lon": -122.267},
        received_monotonic=1.0,
    )
    assert fix is not None
    assert fix.drone == Coordinates(lat=37.522, lon=-122.267)
    assert parse_status({"lat": 0.0, "lon": 0.0}, received_monotonic=1.0) is None


def test_pixel_maps_across_the_camera_footprint() -> None:
    fix = LiveFix(
        drone=Coordinates(lat=37.5, lon=-122.2),
        received_monotonic=0.0,
        top_left=Coordinates(lat=10.0, lon=20.0),
        top_right=Coordinates(lat=10.0, lon=30.0),
        bottom_left=Coordinates(lat=0.0, lon=20.0),
        bottom_right=Coordinates(lat=0.0, lon=30.0),
    )
    center = footprint_at(50, 40, 100, 80, fix)
    assert center.lat == 5
    assert center.lon == 25
    drone_only = LiveFix(drone=fix.drone, received_monotonic=0.0)
    assert footprint_at(10, 10, 100, 80, drone_only) == fix.drone
    state = VisionState(
        frame_id=1,
        timestamp=datetime.now(UTC),
        latency_ms=10.0,
        bounding_boxes=[
            BoundingBox(
                coordinates=[10.0, 10.0, 20.0, 20.0],
                centroid=[20.0, 20.0],
                confidence=0.8,
                classification="hazard",
            )
        ],
        is_occluded=False,
        frame_width=100,
        frame_height=80,
    )
    placed = apply_live_fix(state, drone_only)
    assert placed.drone_coordinates == fix.drone
    assert placed.bounding_boxes[0].geo_coordinates is None
