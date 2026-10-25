# Aegis technical report

Aegis is a maritime and flood search-and-rescue command center. OpenCV 5 ingests a frame, a detector proposes people and boats, and one decision path dispatches a supply payload, reroutes ground rescue, or escalates to a human. The operator sees the same frame, the spatial state, and the action trace. The system records intent. It does not fly a drone or drive a boat.

The diagram is in [architecture.md](architecture.md).

## What is running

Two deployments are distinct.

| Path | Detector | Video | Where it runs |
| --- | --- | --- | --- |
| Local trained evaluation | YOLO11n person/boat, confidence 0.45 | Held-out MOBDrone clips and a local DJI file | This workstation, NVIDIA GeForce RTX 2050 |
| AWS command center | Classical OpenCV detector | One held-out DJI clip inside the image | ECS Fargate, X86_64, 1 vCPU, 2 GB, `ap-southeast-2` |

The cloud image does not contain `best.pt`. `AEGIS_DETECTOR` stays `classical` there. `AEGIS_VIDEO_SOURCE` is one held-out clip, `DJI_0804_0001_30m_1.mp4`, and mock actions are off. The detector in the container emits hazard boxes from that clip, and the missing GPS makes the trace escalate. The public address belongs to the task network interface and changes when the task is replaced. TCP 8000 is limited to the operator address used at deploy time. The HUD labels the host as Fargate, not Graviton, and the cost field stays unconfigured.

`AEGIS_DEPLOYMENT_VERIFIED` remains false. The 124 FPS / 12.4 ms / $0.0031 ribbon in the HUD is an unverified reference printed by the page, not a measurement from this deployment.

## OpenCV 5

`opencv-python-headless` 5.0.0.93 is a runtime dependency, not a version pin in name only.

- `cv2.VideoCapture` acquires files, RTSP, and HLS, including FFmpeg open and read timeouts.
- Preprocessing uses bilateral filtering, CLAHE on the LAB lightness channel, and a Laplacian variance blur test. Variance below 15 forces human review.
- Segmentation builds a coarse background, thresholds the absolute difference, and extracts components with morphology and `connectedComponentsWithStats`.
- Classical detection keeps compact high-contrast person-sized regions and treats larger obstacles as hazards. Warm fragments are debris. Open water alone is not a flood box.
- The bus JPEG-encodes each acquired frame, and the MCP reanalysis path decodes a JPEG with `cv2.imdecode`.

## Data

The detector was trained on a two-class set exported under `data/training/detect`. Those files are gitignored.

| Split | Images | Role |
| --- | --- | --- |
| Train | 9000 | SeaDronesSee Object Detection v2 compressed JPEG train, plus MOBDrone frames |
| Validation | 1568 | SeaDronesSee val, plus MOBDrone clips `DJI_0915_0001`, `DJI_0915_0002`, and `DJI_0915_0003` |
| Held out | DJI_0804 clips | Not present in train or validation |

Class 0 is person, mapped to survivor. Class 1 is boat, mapped to hazard. SeaDronesSee swimmers map to person. Jetski, buoy, life-saving appliances, ignored, surfboard, and life buoy are not trained. MOBDrone wood is debris for scoring only. It is not a model class.

The export totals are 8560 SeaDronesSee training images plus 440 MOBDrone training images, and 1480 SeaDronesSee validation images plus 88 MOBDrone validation images.

## Training

YOLO11n, 20 epochs, image size 640, batch 4, 2 workers, mosaic closed for the last 10 epochs. The run name is `person-boat-sds`. Ultralytics reported 5533 seconds at the end of epoch 20. The device was CUDA on an RTX 2050. Weights stay in `data/training/runs/person-boat-sds/weights/` and are not committed.

Validation-split metrics from `results.csv`:

| Checkpoint | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| Epoch 20 | 0.854 | 0.788 | 0.786 | 0.425 |
| Best mAP50, epoch 18 | 0.850 | 0.789 | 0.790 | 0.426 |
| Best mAP50-95, epoch 19 | 0.858 | 0.789 | 0.790 | 0.430 |

These numbers are the training validation split. They are not the held-out test.

## Held-out test

`scripts/eval_ground_truth.py` and `scripts/eval_trained.py` score every 30th frame of the DJI_0804 clips. The trained run used `best.pt` at confidence 0.45. Both runs scored 1264 frames.

Classical detector:

| Class | IoU | Precision | Recall | Recall in size band | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Person | 0.30 and 0.50 | 0.000 | 0.000 | 0.000 | 0 | 203 | 1546 |
| Boat | 0.30 | 0.701 | 0.404 | 0.624 | 289 | 123 | 426 |
| Boat | 0.50 | 0.626 | 0.361 | 0.574 | 258 | 154 | 457 |
| Wood | 0.30 and 0.50 | 0.000 | 0.000 | 0.000 | 0 | 25 | 329 |

Trained detector at confidence 0.45:

| Class | IoU | Precision | Recall | Recall in size band | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Person | 0.30 | 0.739 | 0.235 | 0.381 | 363 | 128 | 1183 |
| Person | 0.50 | 0.611 | 0.194 | 0.335 | 300 | 191 | 1246 |
| Boat | 0.30 | 0.899 | 0.846 | 0.964 | 605 | 68 | 110 |
| Boat | 0.50 | 0.881 | 0.829 | 0.957 | 593 | 80 | 122 |
| Wood | 0.30 and 0.50 | 0.000 | 0.000 | 0.000 | 0 | 0 | 329 |

Boat detection is the strong result. Person precision is usable and person recall is low: most annotated people are still missed, especially outside the larger size band. Wood is unscored as a class because the model was not trained to emit debris. The classical detector finds no people on this held-out set.

## Decision path

`decide()` reads one `VisionState` and returns one outcome.

- Blur, occlusion, or confidence below 0.65 escalates to a human.
- A verified survivor with a usable geographic fix dispatches a supply payload.
- A hazard or debris box with a usable geographic fix reroutes ground rescue.
- No usable fix escalates. Latitude and longitude of `0,0` are unavailable and are never a dispatch target.
- Real MOBDrone clips have no GPS, so a trained local playback escalates instead of dispatching. The mock source is what demonstrates all three branches with locations. Mock actions are labeled synthetic.

MCP tools are `read_vision`, `assess_frame`, `reanalyze_frame`, `dispatch_payload`, `reroute_ground_rescue`, and `escalate_to_human`. Each action result sets `hardware_executed` to false.

## AWS

The meaningful cloud component is the command center process on ECS.

- Repository `aegis-command-center` in ECR, region `ap-southeast-2`.
- Cluster `aegis`, service `command-center`, task family `aegis-command-center`.
- Fargate platform `X86_64`, 1024 CPU units, 2048 MB.
- Container command is one Uvicorn worker on `0.0.0.0:8000`.
- Same-origin browsers may open `/ws/live`. Other origins still have to match `AEGIS_ALLOWED_ORIGINS`.

Kinesis is not part of this deployment. CloudFormation accepted the template, then `kinesisvideo:CreateStream` was denied by an organization service control policy. The same policy denies `kinesis:CreateStream`. IAM `AdministratorAccess` does not override that deny. The Kinesis plan remains in `infra/aws/` and was not deployed.

Graviton and COOL were not run. No hourly price was entered, so the HUD cost stays unconfigured. The task should be stopped after the demonstration because Fargate is billed while the service is desired.

## Reproduce the local trained path

Dependencies are pinned in `requirements.lock.txt`. Training weights and both datasets stay under `data/` and are not in git.

```bash
uvicorn src.server.main:app --host 127.0.0.1 --port 8000 --workers 1
```

For the trained detector on one held-out clip, set `AEGIS_DETECTOR=trained` and point `AEGIS_VIDEO_SOURCE` at a local `DJI_0804` file. Leave `AEGIS_MOCK_ACTIONS` false for that playback. The cloud task is a separate process and does not need those variables.

## Limitations

- Person recall on DJI_0804 is 0.194 at IoU 0.50. The model is not a complete person detector for small or distant swimmers.
- The AWS page shows the classical detector on one real held-out clip. It does not show the trained weights. The clip has no GPS. The service attaches a synthetic camera pose at 14.601 N, 120.985 E so a boat box can reroute. That pose is not a surveyed rescue location. The live confidence threshold on that task is 0.50.
- There is no live drone stream. Kinesis Video could not be created in this account.
- There is no measured Graviton or COOL result, and there is no measured compute cost.
- The synthetic pose on the AWS clip, and the coordinates on the mock scene, are not a surveyed rescue location. `0,0` is still unavailable.
