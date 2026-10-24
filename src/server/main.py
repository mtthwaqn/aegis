"""Run with uvicorn src.server.main:app --host 127.0.0.1 --port 8000."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.websockets import WebSocketDisconnect

from src.ingestion.video_streamer import CapturedFrame, VideoStreamer
from src.server.bus import BusConflict, IntegrationBus
from src.server.config import Settings
from src.server.schemas import (
    AgentActionPayload,
    ControlState,
    OverrideRequest,
    SystemTelemetry,
    VisionState,
)

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        queue: asyncio.Queue[CapturedFrame] = asyncio.Queue(maxsize=config.frame_queue_size)
        streamer = VideoStreamer(config, queue)
        bus = IntegrationBus(config, streamer)
        application.state.bus = bus
        tasks = [
            asyncio.create_task(streamer.run(), name="video-ingestion"),
            asyncio.create_task(bus.consume_frames(queue), name="frame-encoder"),
            asyncio.create_task(bus.broadcast(), name="live-broadcast"),
        ]
        application.state.tasks = tasks
        if not config.api_token:
            logger.warning("API authentication disabled: use loopback for local evaluation")
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await streamer.close()

    application = FastAPI(title="Aegis Integration Bus", version="1.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Frame-ID", "X-Frame-Timestamp", "X-Frame-Width", "X-Frame-Height"],
    )

    def authorized(token: str | None) -> bool:
        return not config.api_token or (
            token is not None and hmac.compare_digest(token.encode(), config.api_token.encode())
        )

    async def authenticate(authorization: Annotated[str | None, Header()] = None) -> None:
        token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
        if not authorized(token):
            raise HTTPException(
                401, "Valid bearer token required", headers={"WWW-Authenticate": "Bearer"}
            )

    Auth = Depends(authenticate)

    def get_bus(request: Request) -> IntegrationBus:
        return request.app.state.bus

    @application.exception_handler(BusConflict)
    async def conflict_handler(request: Request, error: BusConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @application.middleware("http")
    async def limit_body(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "POST":
            # Bound both Content-Length and chunked bodies before JSON validation.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 256 * 1024:
                    return JSONResponse(status_code=413, content={"detail": "Payload too large"})
            request._body = bytes(body)
        return await call_next(request)

    @application.get("/healthz")
    async def health(request: Request) -> JSONResponse:
        bus = get_bus(request)
        telemetry = await bus.telemetry()
        failed = [task.get_name() for task in request.app.state.tasks if task.done()]
        healthy = not failed and not any(a.alert_id == "source" for a in telemetry.active_alerts)
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "source_status": telemetry.source_status,
                "failed_tasks": failed,
                "schema_version": "1.0",
            },
        )

    @application.get("/command-center", response_class=HTMLResponse)
    async def command_center() -> HTMLResponse:
        return HTMLResponse(
            files("dashboard").joinpath("command_center.html").read_text(encoding="utf-8")
        )

    @application.get("/assets/vision-state.js", include_in_schema=False)
    async def vision_state_script() -> Response:
        return Response(
            files("dashboard").joinpath("vision_state.js").read_text(encoding="utf-8"),
            media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @application.get("/api/telemetry", dependencies=[Auth])
    async def telemetry(request: Request) -> SystemTelemetry:
        return await get_bus(request).telemetry()

    @application.get("/api/control", response_model=ControlState, dependencies=[Auth])
    async def control(request: Request) -> ControlState:
        bus = get_bus(request)
        async with bus.lock:
            return bus.control

    @application.post("/api/control/override", response_model=ControlState, dependencies=[Auth])
    async def override(request: Request, payload: OverrideRequest) -> ControlState:
        return await get_bus(request).override(payload)

    @application.post("/api/telemetry/vision", status_code=202, dependencies=[Auth])
    async def vision(request: Request, state: VisionState) -> dict[str, int | str]:
        await get_bus(request).accept_vision(state)
        return {"status": "accepted", "frame_id": state.frame_id}

    @application.post("/api/agent/action", dependencies=[Auth])
    async def action(request: Request, payload: AgentActionPayload) -> JSONResponse:
        trace = await get_bus(request).record_action(payload)
        return JSONResponse(
            status_code=423 if trace.status == "blocked" else 202,
            content=trace.model_dump(mode="json"),
        )

    @application.get("/api/frames/latest", dependencies=[Auth])
    async def latest(request: Request) -> Response:
        frame = await get_bus(request).latest_frame()
        if frame is None:
            raise HTTPException(503, "No decoded frame available yet")
        return Response(
            frame.jpeg,
            media_type="image/jpeg",
            headers={
                "X-Frame-ID": str(frame.payload.frame_id),
                "X-Frame-Timestamp": frame.payload.timestamp.isoformat(),
                "X-Frame-Width": str(frame.payload.width),
                "X-Frame-Height": str(frame.payload.height),
                "Cache-Control": "no-store",
            },
        )

    @application.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in config.allowed_origins:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        # Credentials travel in the first message, never in query strings/access logs.
        if config.api_token:
            try:
                async with asyncio.timeout(5):
                    message = await websocket.receive_text()
                if len(message) > 2048:
                    raise ValueError("Authentication message too large")
                payload = json.loads(message)
                if (
                    not isinstance(payload, dict)
                    or payload.get("type") != "auth"
                    or not (isinstance(payload.get("token"), str) and authorized(payload["token"]))
                ):
                    raise ValueError("Unauthorized")
            except (ValueError, TimeoutError, WebSocketDisconnect, KeyError):
                with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
                    await websocket.close(code=1008)
                return
        bus: IntegrationBus = websocket.app.state.bus
        try:
            queue = await bus.subscribe()
        except BusConflict:
            await websocket.close(code=1013)
            return

        async def send() -> None:
            while True:
                packet = await queue.get()
                async with asyncio.timeout(2):
                    await websocket.send_text(packet)

        async def receive() -> None:
            while True:
                try:
                    message = await websocket.receive_text()
                except KeyError:
                    await websocket.close(code=1003)
                    return
                if len(message) > 2048:
                    await websocket.close(code=1009)
                    return
                try:
                    event = json.loads(message)
                except ValueError:
                    await websocket.close(code=1007)
                    return
                if not isinstance(event, dict) or event.get("type") != "ping":
                    await websocket.close(code=1008)
                    return

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, TimeoutError, RuntimeError, OSError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            # ASGI servers may cancel the connection during shutdown. Cleanup
            # must finish even under an enclosing AnyIO cancellation scope.
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
                await bus.unsubscribe(queue)
                with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
                    await websocket.close()

    return application


app = create_app()
