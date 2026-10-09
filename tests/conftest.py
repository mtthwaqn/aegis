from datetime import UTC, datetime

import pytest

from src.server.config import Settings
from src.server.schemas import AgentActionPayload, Coordinates, VisionState


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, video_source="mock", mock_actions=False, api_token="")


@pytest.fixture
def action() -> AgentActionPayload:
    return AgentActionPayload(
        action_id="test-action",
        action_type="DISPATCH_SUPPLY_DRONE",
        target_coordinates=Coordinates(lat=14.6, lon=120.98),
        payload_type="medical",
        reasoning_trace="Survivor confirmed → spatial evaluation → supply dispatched",
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def vision() -> VisionState:
    return VisionState(
        frame_id=0,
        timestamp=datetime.now(UTC),
        latency_ms=12.4,
        bounding_boxes=[],
        is_occluded=False,
        frame_width=960,
        frame_height=540,
    )
