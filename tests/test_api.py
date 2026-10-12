from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.server.config import Settings
from src.server.main import create_app
from src.server.schemas import AgentActionPayload


def test_live_api_override_and_validation(settings: Settings, action: AgentActionPayload) -> None:
    with TestClient(create_app(settings)) as client:
        with client.websocket_connect("/ws/live") as ws:
            live = ws.receive_json()
            for _ in range(30):
                if live["frame"]:
                    break
                live = ws.receive_json()
            assert live["frame"]["frame_id"] == live["vision"]["frame_id"]
            assert client.post("/api/telemetry/vision", json=live["vision"]).status_code == 202
            unknown = live["vision"] | {"frame_id": 999999}
            assert client.post("/api/telemetry/vision", json=unknown).status_code == 409
            assert client.get("/healthz").status_code == 200
            assert client.get("/api/frames/latest").headers["content-type"] == "image/jpeg"
            assert (
                client.post("/api/agent/action", json=action.model_dump(mode="json")).status_code
                == 202
            )
            assert (
                client.post(
                    "/api/control/override",
                    json={"enabled": True, "reason": "Commander took control"},
                ).status_code
                == 200
            )
            blocked = action.model_copy(update={"action_id": "blocked"}).model_dump(mode="json")
            response = client.post("/api/agent/action", json=blocked)
            assert response.status_code == 423
            assert response.json()["status"] == "blocked"
            assert (
                client.post("/api/telemetry/vision", json={"frame_id": "invalid"}).status_code
                == 422
            )
            assert (
                client.post(
                    "/api/agent/action",
                    content="{broken",
                    headers={"Content-Type": "application/json"},
                ).status_code
                == 422
            )
            assert client.post("/api/agent/action", content="x" * 270000).status_code == 413
            telemetry = client.get("/api/telemetry").json()
            assert telemetry["control"]["override_enabled"]
            assert telemetry["estimated_cost_usd_per_100k_frames"] is None
            assert not telemetry["deployment_verified"]
        assert client.get("/command-center").status_code == 200
        assert client.get("/api/telemetry").json()["connected_viewers"] == 0


def test_auth_and_origin_checks(settings: Settings, action: AgentActionPayload) -> None:
    secured = settings.model_copy(update={"api_token": "secret-test-token"})
    with TestClient(create_app(secured)) as client:
        assert client.get("/api/telemetry").status_code == 401
        assert (
            client.post("/api/agent/action", json=action.model_dump(mode="json")).status_code == 401
        )
        assert (
            client.get(
                "/api/telemetry", headers={"Authorization": "Bearer secret-test-token"}
            ).status_code
            == 200
        )
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "auth", "token": "secret-test-token"})
            assert ws.receive_json()["type"] == "live"
        import pytest

        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "auth", "token": "incorrect"})
            ws.receive_json()
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/live", headers={"origin": "https://evil.test"}):
                pass


def test_invalid_websocket_messages_close_cleanly(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        for content, expected_code in [("{invalid", 1007), ('{"type":"unknown"}', 1008)]:
            with client.websocket_connect("/ws/live") as ws:
                ws.send_text(content)
                while True:
                    message = ws.receive()
                    if message["type"] == "websocket.close":
                        assert message["code"] == expected_code
                        break
        with client.websocket_connect("/ws/live") as ws:
            ws.send_bytes(b"binary-command")
            while True:
                message = ws.receive()
                if message["type"] == "websocket.close":
                    assert message["code"] == 1003
                    break
        assert client.get("/api/telemetry").json()["connected_viewers"] == 0
