# Aegis Command Center — Member 1

Python 3.11+ contracts, video ingestion, an asynchronous FastAPI integration bus,
and a responsive Dark Ops operator dashboard for maritime / flood search and rescue.
Members 2–5 supply perception, georeferencing, physical MCP tools, AWS deployment,
and COOL benchmarking. No hardware commands execute in this repository.

PR #3's Member 3 reasoning is now connected to the live Command Center. Each
accepted `VisionState` passes through `to_reasoning_payload()` and a validated
`ReasoningState`, then travels in `/ws/live` alongside its matching frame. The
**Live Spatial State** panel displays target, confidence, hazard, location, and
current state for the frame actually rendered in the video viewport.

## Setup and run

```bash
bash scripts/setup.sh
source .venv/bin/activate
uvicorn src.server.main:app --host 127.0.0.1 --port 8000 --workers 1 \
  --ws-max-size 4096 --ws-max-queue 4 --timeout-graceful-shutdown 10
```

Open **http://localhost:8000/command-center**. OpenAPI is at
**http://localhost:8000/docs**. The optional Streamlit launcher runs in another terminal:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Both entry points use the same browser WebSocket renderer. Video updates at 25 Hz
independently of Streamlit reruns. The default `mock` source generates a paced
30 FPS scene, georeferenced synthetic detections, periodic occlusion, and three-second
action branches: survivor → supply dispatch, hazard → convoy diversion, degraded
vision → human escalation. Mock events carry `origin="mock"` and never operate hardware.

The ribbon shows **measured** acquisition / vision FPS and rolling p95 vision
latency. The supplied 124 FPS / 12.4 ms / $0.0031 per 100k frames are explicitly
labeled unverified benchmark targets. AWS compute is a configured target until
Member 5 verifies deployment. Cost stays unconfigured until a verified hourly
compute price is supplied; the estimate excludes network, storage, Kinesis,
and other services. COOL benchmark results are not fabricated.

## Architecture and team boundaries

```text
src/ingestion/video_streamer.py   paced acquisition + single-owner capture thread
src/server/schemas.py             strict Pydantic v2 shared contracts
src/server/config.py              validated environment settings
src/server/bus.py                 bounded correlation, metrics, safety, audit, fanout
src/server/main.py                FastAPI lifecycle, HTTP API, WebSocket gateway
dashboard/app.py                  Streamlit launcher
dashboard/command_center.html    responsive browser HUD + interactive geographic grid
scripts/setup.sh                 reproducible installation
scripts/export_schemas.py        standalone JSON Schema export
scripts/post_vision.py           continuous classical-perception publisher
tests/                           contracts, ingestion, safety, overload, API integration
```

The dependency lock includes **OpenCV 5** for ingestion. Member 2's preprocessing,
segmentation, target detection, and hazard detection algorithms remain separate.
Member 3 preserves frame identity and provides geographic fixes. Member 4 reports
execution events and enforces commander authority at the physical tool boundary.

The bus runs in **one Uvicorn worker**. Decoder operations run on a dedicated
thread; queue / state access stays on the asyncio loop. JPEG encoding runs off-loop
once per acquired frame. A bounded acquisition queue discards oldest frames under
load. Each of at most 64 viewers has a one-snapshot queue; stale snapshots coalesce
and stalled sends time out. Multiple workers require a shared broker / state store.

Reasoning runs once per accepted vision frame and shares its bounded cache lifetime.
`LiveEnvelope.reasoning` is null when the selected video frame has no matching
vision. Existing vision POST bodies are unchanged; publishers send raw `VisionState`,
not the standalone reasoning CLI's `{vision, reasoning, georeferencing}` wrapper.
Clients that validate WebSocket messages with `extra="forbid"` must regenerate their
`LiveEnvelope` schema to recognize the new optional `reasoning` field.

## Rubric evidence

| Competition target | Member 1 evidence |
| --- | --- |
| Technical execution (30%) | Validated contracts, off-loop decoding / encoding, bounded queues, synchronized delivery |
| Innovation (20%) | Reproducible perception → spatial evaluation → dispatch / reroute / escalation branches; physical execution provided by Member 4 |
| Real-world impact (20%) | Operator visibility, uncertainty alerts, freshness indicators, and reduced monitoring burden |
| User experience (10%) | Live HUD, trace terminal, tactical interaction, responsive layout, commander override |
| Documentation (10%) | Monorepo boundaries, installation lock, local mocks, OpenAPI, exported schemas |
| Cloud / responsibility (10%) | Kinesis adapter, explicit compute / cost provenance, bounded audit history, human-review latch; Graviton / COOL verification provided by Member 5 |

## Environment configuration

Setup copies `.env.example` to `.env` only if none exists. Backend settings read
`.env`; process variables take precedence. Streamlit reads its backend URL from
the process environment.

| Variable | Default | Meaning |
| --- | --- | --- |
| `AEGIS_VIDEO_SOURCE` | `mock` | Path, RTSP(S), HTTP(S) HLS URL, or `kinesis://stream-name` |
| `AEGIS_INGESTION_FPS` | `30` | Maximum acquisition rate; no burst catch-up |
| `AEGIS_BROADCAST_HZ` | `25` | Snapshot rate, 1–30 Hz |
| `AEGIS_MOCK_ACTIONS` | `true` | Action simulation only for built-in mock video |
| `AEGIS_FRAME_QUEUE_SIZE` | `2` | Bounded capture queue |
| `AEGIS_FRAME_CACHE_SIZE` | `120` | Correlation cache, roughly four seconds at 30 FPS |
| `AEGIS_VISION_WAIT_MS` | `150` | Wait for matching vision before sending video alone |
| `AEGIS_JPEG_QUALITY` | `80` | JPEG quality, 20–95 |
| `AEGIS_READ_TIMEOUT_MS` | `3000` | Network FFmpeg open / read timeout |
| `AEGIS_SOURCE_STALE_SECONDS` | `5` | Source / telemetry freshness threshold |
| `AEGIS_LOW_CONFIDENCE_THRESHOLD` | `0.65` | Automatic human-review threshold |
| `AEGIS_API_TOKEN` | empty | Loopback evaluation mode; configure a secret for remote access |
| `AEGIS_ALLOWED_ORIGINS` | localhost 8000 / 8501 | JSON array of exact trusted browser origins |
| `AEGIS_COMPUTE_INSTANCE` | `AWS Graviton3 c7g.xlarge + COOL` | Declared compute target |
| `AEGIS_DEPLOYMENT_VERIFIED` | `false` | Set after actual deployment verification |
| `AEGIS_HOURLY_COMPUTE_COST_USD` | unset | Verified hourly compute cost for estimates |
| `AEGIS_KINESIS_REGION` | `ap-southeast-2` | AWS region |
| `AEGIS_KINESIS_DATA_ENDPOINT` | unset | Optional data API endpoint, not a playlist URL |
| `AEGIS_KINESIS_SESSION_SECONDS` | `3600` | Signed HLS session lifetime, renewed before expiry |
| `AEGIS_DASHBOARD_BACKEND_URL` | `http://localhost:8000` | Browser-reachable Streamlit iframe backend |
| `AEGIS_DETECTOR` | `classical` | `trained` selects the person/boat weights. Unset stays classical |
| `AEGIS_DETECTOR_WEIGHTS` | `data/training/runs/person-boat-sds/weights/best.pt` | Local weights used only when the detector is `trained` |
| `AEGIS_DETECTOR_CONF` | `0.45` | Minimum score kept by the trained detector |

Local clips loop automatically:

```bash
AEGIS_VIDEO_SOURCE=data/mobdrone.mp4 AEGIS_MOCK_ACTIONS=false \
  uvicorn src.server.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Then run the live publisher in a second terminal from the repository root:

```bash
source .venv/bin/activate
python scripts/post_vision.py --interval 0.1
```

The publisher reads `AEGIS_API_TOKEN` from the environment / `.env` and authenticates
both acquisition and vision POST requests. It continuously publishes survivor,
hazard, low-confidence, and empty-scene results until Ctrl-C; it does not stop on
the first person. Use `--once` for one frame or `--attempts 20` for a bounded run.
Connection / decode failures retry; authentication failures exit. A 409 indicates
an expired or conflicting immutable frame, so the next attempt acquires a new frame.
The built-in `mock` feed already supplies its own vision; use a local clip / RTSP
source for this publisher to avoid conflicting with synthetic detections.

The live `analyze()` path is locked to the classical detector unless the publisher process
sets `AEGIS_DETECTOR=trained`. Trained mode keeps person boxes as `survivor` and boat
boxes as `hazard` at confidence 0.45. Weights on disk do not switch the path, and
geographic coordinates stay empty. Install the optional package first with
`pip install ultralytics`. The separate 0.65 human-review threshold is applied by
Member 3 reasoning using the server's `AEGIS_LOW_CONFIDENCE_THRESHOLD` configuration.

```bash
AEGIS_DETECTOR=trained python scripts/post_vision.py --interval 0.1
```

Frames are scaled to fit 1280×720 while preserving aspect ratio. Member 2/3 must
use the **decoded dimensions** returned by the frame endpoint. The reader paces
frame reads; it does not interpolate clips with a different native FPS. RTSP
credentials belong in `.env`, not shell arguments. Connection drops / decode
errors retry with 0.25–5 second exponential backoff and degraded health status.
A missing local path fails startup. An unreadable source retries.

Kinesis uses boto3's credential chain / IAM role:

```bash
AEGIS_VIDEO_SOURCE=kinesis://rescue-drone AEGIS_KINESIS_REGION=ap-southeast-2 \
  uvicorn src.server.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The reader IAM policy allows `kinesisvideo:DescribeStream`,
`kinesisvideo:GetDataEndpoint`, and `kinesisvideo:GetHLSStreamingSessionURL`
on `aegis-rescue`. Playlist and fragment reads use that signed HLS URL; they are
not separate IAM actions. Account setup, the producer policy, and the stream
plan live in [`infra/aws/README.md`](infra/aws/README.md). That plan does not
deploy the app or record a benchmark. A configured data endpoint skips discovery.
Signed HLS URLs never enter telemetry.
The stream requires retained fragments and compatible HLS video codecs. HLS adds
fragment latency; the bus rate does not guarantee low-latency Kinesis acquisition.
See the [AWS HLS API requirements](https://docs.aws.amazon.com/kinesisvideostreams/latest/APIReference/API_reader_GetHLSStreamingSessionURL.html).

## Integration API

When configured, protected HTTP calls require `Authorization: Bearer <token>`.

| Endpoint | Behavior |
| --- | --- |
| `GET /healthz` | 200 healthy source / tasks; 503 starting, stale, reconnecting, or failed |
| `GET /api/frames/latest` | JPEG with `X-Frame-ID`, `X-Frame-Timestamp`, `X-Frame-Width`, `X-Frame-Height` |
| `POST /api/telemetry/vision` | Strict `VisionState`: 202 accepted, 409 correlation conflict, 422 invalid contract |
| `POST /api/agent/action` | Strict `AgentActionPayload`: 202 accepted / escalated, 423 blocked, 409 conflicting reused ID |
| `GET /api/telemetry` | `SystemTelemetry`: measured metrics, alerts, 50-entry decision ring |
| `GET /api/control` | Current commander latch |
| `POST /api/control/override` | `{ "enabled": true, "reason": "Commander intervention" }` |
| `WS /ws/live` | `LiveEnvelope`: frame, its matching vision, telemetry including traces |

All models forbid unknown fields. Numbers and booleans reject string coercion;
confidence is `[0,1]`; NaN / infinity are rejected. Timestamps require a timezone.
Bounding boxes use pixel `[x,y,w,h]` with positive dimensions, centroid `[cx,cy]`
consistent within half a pixel, and bounds contained in the decoded frame.
Classifications: `survivor`, `hazard`, `debris`. POST bodies are capped at 256 KiB;
collections and text fields are bounded.

### Members 2 / 3: frame correlation

Fetch `/api/frames/latest`, process its JPEG, and preserve the exact frame ID and
timestamp headers. POST the spatial state before cache expiry. Local / RTSP feeds
await external vision; the mock feed already publishes immutable synthetic vision.
Unknown / expired IDs, differing decoded dimensions, conflicting duplicate states,
or timestamps differing from the captured frame by over one second return 409.
Out-of-order vision is cached without rewinding safety evaluation.

The panel preserves Member 3's state precedence: `LOW_CONFIDENCE`, then
`HAZARD_DETECTED`, then `TARGET_VERIFIED`, otherwise `NO_TARGET`. The confidence
card shows the lowest detection score across the frame; the target card separately
shows the selected survivor's confidence. Hazard presence remains visible even
when the overall state is low confidence. Empty scenes and missing vision are
displayed distinctly. Stale or disconnected feeds clear the five current fields.

Location describes Member 3's selected detection, which may be a hazard. Missing
detection GPS displays **Location unavailable**, even if drone GPS is known; drone
position is labeled separately and is never substituted for a survivor fix. Enrich
geographic coordinates before the first POST: accepted frame states are immutable,
so a second POST adding GPS to the same frame returns 409.

```python
from datetime import datetime
from src.server.schemas import BoundingBox, VisionState

state = VisionState(
    frame_id=7,
    timestamp=datetime.fromisoformat("2026-10-08T15:00:00Z"),
    latency_ms=12.4,
    frame_width=960,
    frame_height=540,
    is_occluded=False,
    bounding_boxes=[
        BoundingBox(
            coordinates=[100.0, 140.0, 40.0, 60.0],
            centroid=[120.0, 170.0],
            confidence=0.94,
            classification="survivor",
        )
    ],
)
print(state.model_dump_json())
```

The example illustrates the contract; use acquired headers and actual detections
for a POST. Optional `geo_coordinates` on boxes and `drone_coordinates` on vision
must come from Member 3 georeferencing, never a guessed pixel-to-geographic mapping.
Optional agent `route_coordinates` supplies up to 128 actual geographic route points.

The broadcaster selects the newest correlated frame, or a frame older than the
configured wait budget. Only that frame's vision is attached; unmatched frames
send `vision=null`. The UI shows video-only status rather than attaching old boxes.

### Member 4: autonomous event example

```bash
curl -X POST http://localhost:8000/api/agent/action \
  -H 'Content-Type: application/json' \
  -d '{
    "action_id": "mission-001-supply-001",
    "action_type": "DISPATCH_SUPPLY_DRONE",
    "target_coordinates": {"lat": 14.6006, "lon": 120.9833},
    "payload_type": "flotation_and_medical",
    "reasoning_trace": "Survivor confirmed -> spatial evaluation -> action dispatched",
    "timestamp": "2026-10-08T15:00:00Z"
  }'
```

Supported actions: `DISPATCH_SUPPLY_DRONE`, `REROUTE_CONVOY`, `ESCALATE_HUMAN`.
`source_frame_id` is optional. Stable `action_id` values provide idempotency;
otherwise a UUID is generated. The last 1,000 IDs are remembered independently
of the 50-event ring. Retrying an earlier accepted action returns its original
receipt, which is **not permission to execute hardware again**.

The local agent turns one `VisionState` into at most one of those actions:

| Vision | Tool | Wire action |
| --- | --- | --- |
| Verified survivor with a geographic fix | `dispatch_payload` | `DISPATCH_SUPPLY_DRONE` / `rescue_payload` |
| Hazard with a geographic fix | `reroute_ground_rescue` | `REROUTE_CONVOY` / `ground_rescue` |
| Low confidence, blur, occlusion, or a missing fix | `escalate_to_human` | `ESCALATE_HUMAN` |
| No target | none | no action |

`0,0` remains the unavailable coordinate and is never a dispatch or reroute target.
MCP tools `read_vision` and `assess_frame` run first. Degraded frames call
`reanalyze_frame` once. Blur and occlusion are not cleared by that second look.
The execution trace is the tool list stored in `reasoning_trace` and shown in the
command-center terminal. `hardware_executed` stays false.

```bash
python scripts/detect_and_act.py
python -m src.mcp.server
```

Docker, ECR, and ECS definitions live in [`infra/runtime/`](infra/runtime/README.md)
and are not deployed. `scripts/benchmark_local.py` times the locked classical
detector on the current machine. It does not record a Graviton or COOL number.

### WebSocket authentication and browser UI

When a token is configured, send `{"type":"auth","token":"secret"}` as the
first WebSocket message within five seconds. With local authentication disabled,
omit this message. Tokens never enter URLs / access logs. Browser origins are
checked separately. Optional later `{"type":"ping"}` messages are accepted;
snapshots serve as the heartbeat. Invalid JSON / unsupported messages close the
connection with a protocol error.

`LiveEnvelope` carries `type="live"`, a monotonic publication `sequence`, a nullable
`frame` with JPEG base64 and acquisition metadata, nullable matching `vision`, and
nullable matching `reasoning` with target, confidence, hazard, location, and spatial
state, plus complete `telemetry`. All three frame IDs must agree when reasoning is
present. Run the schema exporter below for the exact JSON definitions.

The UI reconnects with bounded backoff and flags stale video. It shows survivor
boxes, box-derived translucent hazard masks, terminal-style reasoning traces,
and an interactive geographic grid with pan, zoom, recenter, and marker selection.
The map draws georeferenced detections and explicitly supplied diversion routes.
This offline grid does not claim a satellite basemap, and its hazard overlays do
not claim semantic segmentation polygons.

## Human control and responsible operation

The override button requests a server-side latch and displays confirmation only
after the backend succeeds. Future non-escalation submissions return 423 and are
logged as blocked. Human escalations remain active. Resume requires a deliberate
UI confirmation. Entering a low-confidence / occluded episode automatically creates
an `ESCALATE_HUMAN` event and active alert; repeated degraded frames do not flood the
log. Recovery clears the alert. For escalations lacking a geographic fix, drone
position is used when known, otherwise `0,0` explicitly means unavailable and must
not be used as a dispatch target.

Member 4 **must enforce the latch at the physical command boundary** and implement
flight-controller cancellation for existing missions. A status check followed by
execution is not an atomic hardware interlock. The bus cannot undo commands already
executed by another service; its event receipts do not confirm hardware completion.

Audit / idempotency / frame caches are bounded, process-local, and reset on restart.
Member 5 must add persistent audit storage for durable retention. Ctrl-C cancels
background tasks, closes sockets, and releases capture on its owner thread after
pending reads complete. For remote deployments, configure a secret token, exact
HTTPS origins, TLS termination, and a browser-reachable HTTPS dashboard backend URL.

## Verification and reproducibility

```bash
source .venv/bin/activate
pytest -q
ruff check .
python scripts/export_schemas.py
python scripts/smoke_test.py
curl http://localhost:8000/healthz
```

With Node.js available, run the dependency-free dashboard presentation checks:

```bash
node --test tests/vision_state.test.cjs
```

Tests exercise strict contracts, geometry, real MP4 loops, pacing, bounded backlog,
network-drop recovery, synchronized WebSocket delivery, low-confidence escalation,
override / resume, idempotency, audit retention, auth / origin checks, body size
limits, and malformed JSON. Real RTSP / AWS interoperability requires live devices
and credentials; local tests do not certify those deployments.

The integration tests also exercise authenticated vision POST → typed reasoning →
correlated WebSocket delivery for verified survivors, low confidence, hazards,
and empty frames. Presentation tests verify all five fields, unknown locations,
stale / mismatched frame clearing, and the distinction between drone and detection GPS.

`requirements.lock.txt` pins tested dependencies; `pyproject.toml` defines compatible
ranges. Local verification used Python 3.13.7 and OpenCV 5.0.0; all pinned packages
declare Python 3.11 compatibility. The mock smoke test delivered 25.0 Hz with 29.9
FPS acquisition. These measurements include no real inference workload.

Member 5 must record the real Graviton instance, input resolution, OpenCV
build, inference workload, sustained FPS, p95, and COOL measurements. Local mock
throughput is not an inference or Graviton benchmark. Integration follows
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) and
[WebSocket support](https://fastapi.tiangolo.com/advanced/websockets/).
