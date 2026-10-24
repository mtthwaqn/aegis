# Member 5 — AWS account, IAM, and Kinesis plan

This is the current cloud slice. The OpenCV workload already on `main` stays as it is.
Do not deploy the application, and do not benchmark. The detector will change when
training lands, so this plan carries video only and does not pin a model.

## Account

Use one shared team account in **ap-southeast-2**. That matches `AEGIS_KINESIS_REGION`.

1. Turn on MFA for the root user, then stop using root for daily work.
2. Create one IAM user for this slice, for example `aegis-member5`.
3. Attach `iam/member5-bootstrap-policy.json` to that user. It can manage the
   `aegis-rescue` stream, the `aegis-kinesis-*` policies, and the `aegis-kinesis`
   CloudFormation stack. It cannot create ECS, ECR, EC2, or other application resources.
4. Install AWS CLI v2 and authenticate on your machine with `aws configure` or
   `aws configure sso`. Confirm with `aws sts get-caller-identity`.
5. Never commit access keys, and never put `AWS_ACCESS_KEY_ID` or
   `AWS_SECRET_ACCESS_KEY` in a source file. `.env` is gitignored.

## IAM

| Policy | Who uses it | What it allows |
| --- | --- | --- |
| `iam/member5-bootstrap-policy.json` | Member 5, while setting up | Create and describe `aegis-rescue`, and manage this plan's policies and stack |
| `iam/kinesis-reader-policy.json` | The process that already calls Kinesis in `src/ingestion/video_streamer.py` | `DescribeStream`, `GetDataEndpoint`, `GetHLSStreamingSessionURL` |
| `iam/kinesis-producer-policy.json` | The camera or UAV publisher, when that side exists | `DescribeStream`, `GetDataEndpoint`, `PutMedia` |

Playlist and fragment downloads (`GetHLSMasterPlaylist`, `GetHLSMediaPlaylist`,
`GetMP4MediaFragment`) ride the signed HLS URL returned by
`GetHLSStreamingSessionURL`. They are not separate IAM actions. The reader renews
that URL 60 seconds before `AEGIS_KINESIS_SESSION_SECONDS` expires. Signed URLs
stay out of telemetry and out of git.

Attach the reader policy to the local caller only when you are ready to try
`kinesis://aegis-rescue`. Do not create an EC2 instance role in this slice.

## Kinesis plan

Apply this only when the account above exists. It creates the stream and the two
runtime policies. It does not start the app.

```bash
aws cloudformation deploy \
  --region ap-southeast-2 \
  --stack-name aegis-kinesis \
  --template-file infra/aws/kinesis-stream.yaml \
  --capabilities CAPABILITY_IAM
```

| Setting | Value | Why |
| --- | --- | --- |
| Stream | `aegis-rescue` | `AEGIS_VIDEO_SOURCE=kinesis://aegis-rescue` |
| Region | `ap-southeast-2` | Same as `AEGIS_KINESIS_REGION` |
| Media | `video/h264` | The existing OpenCV FFmpeg reader consumes HLS |
| Retention | 24 hours (1–168 allowed) | Live HLS needs retained fragments |
| Playback | `LIVE`, server timestamp | Already how `VideoStreamer` requests the session |

The stack refuses every region except `ap-southeast-2`. After a later apply, leave
the app on `AEGIS_VIDEO_SOURCE=mock` until a real publisher is sending H.264.
`AEGIS_DEPLOYMENT_VERIFIED` stays `false`. No hourly cost is recorded here.

## Explicitly later

- Docker, ECR, ECS, and deploying the command center
- Graviton, COOL, and any FPS / latency / cost benchmark
- Swapping the detector when training arrives; that change belongs in perception, not in this stream
