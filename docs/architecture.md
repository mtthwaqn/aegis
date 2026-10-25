# Aegis architecture

Aegis turns one video frame into a rescue decision and shows that decision to an operator. OpenCV 5 acquires and prepares the frame. A detector writes a `VisionState`. `decide()` picks at most one action. The command center renders the frame, the spatial state, and the action trace. No tool moves hardware.

```mermaid
flowchart TD
  source["Video source<br/>mock, file, RTSP, or HLS"]
  opencv["OpenCV 5<br/>capture, preprocess, segment, JPEG"]
  classical["Classical detector<br/>default live path"]
  trained["YOLO11n person and boat<br/>only when AEGIS_DETECTOR=trained"]
  vision["VisionState"]
  spatial["Spatial state"]
  decision["decide<br/>dispatch, reroute, or escalate"]
  mcp["MCP tools<br/>read, assess, reanalyze, act"]
  bus["Integration bus<br/>one Uvicorn worker"]
  hud["Command center<br/>/command-center and /ws/live"]

  source --> opencv
  opencv --> classical
  opencv --> trained
  classical --> vision
  trained --> vision
  vision --> spatial --> decision
  vision --> mcp
  decision --> mcp
  vision --> bus
  mcp --> bus
  bus --> hud
```

The live default stays classical. Setting `AEGIS_DETECTOR=trained` selects local weights at `data/training/runs/person-boat-sds/weights/best.pt` with confidence 0.45. Those weights are not in git and are not in the cloud image.

```mermaid
flowchart LR
  image["Local image<br/>classical detector, mock video"]
  ecr["ECR<br/>aegis-command-center"]
  fargate["ECS Fargate X86_64<br/>1 vCPU, 2 GB<br/>ap-southeast-2"]
  operator["Operator browser<br/>TCP 8000"]

  image --> ecr --> fargate --> operator
```

Kinesis Video is a separate plan in `infra/aws/kinesis-stream.yaml`. The organization service control policy denies `kinesis` and `kinesisvideo`, so that stream was not created. The running task does not read a drone stream. Graviton and COOL were not measured. The task is x86 Fargate, and `AEGIS_DEPLOYMENT_VERIFIED` stays false.

A missing geographic fix, including coordinates `0,0`, never becomes a dispatch or reroute target. Blur and occlusion force human review. `hardware_executed` is always false.
