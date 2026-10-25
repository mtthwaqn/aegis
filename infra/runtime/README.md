# Runtime image

This image is what the Sydney Fargate service runs. It is X86_64, not Graviton,
and it does not measure COOL. `AEGIS_DEPLOYMENT_VERIFIED` stays `false`.

The container plays one held-out clip, `DJI_0804_0001_30m_1.mp4`, and
`scripts/post_vision.py` runs the classical OpenCV detector on those frames.
Mock actions stay off, so the trace comes from the detector. The clip has no
GPS, so the decision escalates. Copy that clip to `media/clip.mp4` before
building. The clip is not committed.

Do not label this task as Graviton or COOL.
