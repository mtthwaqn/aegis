# Runtime image

This image is what the Sydney Fargate service runs. It is X86_64, not Graviton,
and it does not measure COOL. `AEGIS_DEPLOYMENT_VERIFIED` stays `false`.

The container plays one held-out clip, `DJI_0804_0001_30m_1.mp4`, and
`scripts/post_vision.py` runs the classical OpenCV detector on those frames.
Mock actions stay off, so the trace comes from the detector. The clip has no
GPS. `AEGIS_SYNTHETIC_GEO` places each box on a demo camera pose at 14.601 N,
120.985 E, 30 m above the water. That pose is not a surveyed fix. The live
threshold is 0.50 so a classical boat box can reroute; weaker or occluded
frames still escalate, and 0,0 stays invalid. Playback is 1 frame per second
so the detector can attach to the frame on screen. Copy that clip to `media/clip.mp4` before
building. The clip is not committed.

Do not label this task as Graviton or COOL.
