# Aegis architecture

## 1. From video to a detection

This is the perception path. OpenCV 5 takes one video source and prepares the frame. The live detector stays classical unless `AEGIS_DETECTOR=trained`. The trained path is YOLO11n for person and boat only, at confidence 0.45, and its weights stay on the workstation. Both paths end in one `VisionState`. The dotted Kinesis arrow is a planned input. That stream was not created.

```mermaid
flowchart LR
  subgraph input [Video input]
    mock[Mock scene]
    file[Video file]
    rtsp[RTSP]
    hls[HLS]
    kinesis[Kinesis Video plan<br/>not created]
  end

  subgraph opencv [OpenCV 5]
    capture[VideoCapture<br/>FFmpeg timeouts]
    resize[Resize]
    prep[Bilateral filter<br/>CLAHE]
    blur[Laplacian variance<br/>blur under 15]
    segment[Background difference<br/>connected components]
    jpeg[JPEG encode]
  end

  subgraph perception [Perception]
    choice{AEGIS_DETECTOR}
    classical[Classical detector<br/>person, hazard, debris]
    trained[YOLO11n<br/>person and boat<br/>confidence 0.45]
    weights[best.pt<br/>this PC only]
    vision[VisionState]
  end

  mock --> capture
  file --> capture
  rtsp --> capture
  hls --> capture
  kinesis -.-> capture
  capture --> resize --> prep
  prep --> blur
  prep --> segment
  resize --> jpeg
  segment --> classical
  capture --> choice
  choice -->|default| classical
  choice -->|trained| trained
  weights --> trained
  classical --> vision
  trained --> vision
```

## 2. From a detection to one action

`decide()` returns at most one outcome. A blurred frame, an occluded frame, or confidence below 0.65 always goes to a human. A survivor can dispatch a supply payload, and a hazard can reroute ground rescue, only when that box has a real location. Missing coordinates and `0,0` are not locations, so those frames escalate. A frame with no person and no hazard holds.

```mermaid
flowchart LR
  vision[VisionState] --> spatial[Spatial state]
  blur[Blur] -->|variance under 15| escalate[Escalate to human]
  spatial -->|occlusion or confidence under 0.65| escalate
  spatial -->|no target| hold[Hold]
  spatial -->|survivor or hazard| geo{Usable geographic fix}
  geo -->|survivor with a fix| dispatch[Dispatch supply payload]
  geo -->|hazard or debris with a fix| reroute[Reroute ground rescue]
  geo -->|missing or 0,0| escalate
```

## 3. From the action to the operator screen

The MCP tools record the decision. `hardware_executed` stays false, so nothing here flies a drone or drives a boat. One Uvicorn worker matches each JPEG frame to its `VisionState` and pushes the pair over `/ws/live`. The command center draws the video, the reasoning trace, and the tactical view. The commander can latch an override and block later autonomous submissions. Mock actions add synthetic locations. That path is what the AWS page uses to show dispatch, reroute, and escalate.

```mermaid
flowchart LR
  subgraph mcp [MCP tools]
    read[read_vision]
    assess[assess_frame]
    reanalyze[reanalyze_frame]
    act[dispatch_payload<br/>reroute_ground_rescue<br/>escalate_to_human]
    hardware[hardware_executed<br/>always false]
  end

  subgraph bus [Integration bus]
    queue[Bounded frame queue]
    correlate[Match frame to vision<br/>150 ms]
    worker[One Uvicorn worker<br/>port 8000]
    live["/ws/live"]
    http[HTTP API<br/>override latch]
    mockActions[Mock actions<br/>synthetic locations]
  end

  subgraph hud [Command center]
    feed[Perception HUD]
    traces[Reasoning traces]
    tactical[Tactical view]
    override[Human commander override]
  end

  vision2[VisionState] --> read
  blur2[Blur] --> assess
  jpeg[JPEG frame] --> reanalyze
  jpeg --> queue --> worker
  vision2 --> correlate --> worker
  act --> hardware --> worker
  mockActions --> worker
  worker --> live
  worker --> http
  live --> feed
  live --> traces
  live --> tactical
  override --> http
```

## 4. Where the two demos run

These are separate processes. The local page plays a held-out DJI clip through the trained model. Those clips have no GPS, so the trace escalates. The AWS page is the container on Fargate in `ap-southeast-2`: classical detector, synthetic scene, TCP 8000 open only to the operator address. It does not contain the trained weights.

```mermaid
flowchart LR
  subgraph local [Local trained playback]
    dji[DJI_0804 clips<br/>no GPS]
    post[post_vision.py<br/>coordinates empty]
    localHud[127.0.0.1:8000<br/>escalate only]
  end

  subgraph aws [AWS ap-southeast-2]
    image[Container<br/>classical and mock]
    ecr[ECR aegis-command-center]
    cluster[ECS cluster aegis]
    fargate[Fargate X86_64<br/>1 vCPU, 2 GB]
    guard[TCP 8000<br/>operator address only]
    site[Public command center<br/>synthetic feed]
  end

  dji --> post --> localHud
  image --> ecr --> cluster --> fargate --> guard --> site
```
