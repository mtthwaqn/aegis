# Runtime image

This image is what the Sydney Fargate service runs. It is X86_64, not Graviton,
and it does not measure COOL. `AEGIS_DEPLOYMENT_VERIFIED` stays `false`.

The container plays the public Skydio demo, `rtsps://stream.skydio.com/demo/Skydio/color`,
and reads GPS from `wss://stream.skydio-dev.com/data/demo`. No token is required.
`scripts/post_vision.py` runs the classical OpenCV detector and attaches that
feed's drone fix. A box uses the camera footprint when Skydio sends one.
Mock actions stay off. This GPS is the demo aircraft's fix, not a surveyed
victim location, and it is not frame-accurate. `0,0` stays invalid. The live
threshold is 0.50. Playback stays at 8 frames per second, and the page keeps
the newest frame that already has a detection. The held-out clip can still be
copied to `media/clip.mp4` before building, but it is not the configured source
and it is not committed.

Do not label this task as Graviton or COOL.
