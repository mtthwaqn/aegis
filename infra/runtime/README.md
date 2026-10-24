# Runtime image plan

This directory defines a local Docker image and an ECS task shape. It does not
deploy the app. Do not `docker push` to ECR, and do not call
`ecs:RegisterTaskDefinition`.

The live detector in the image is classical. `AEGIS_DEPLOYMENT_VERIFIED` stays
`false`. The video source stays `mock` until a real publisher is sending H.264
to the Kinesis stream planned in [`../aws/README.md`](../aws/README.md). That
stream is a separate plan and is not created by these files.

The task definition names ARM64 because the intended later host is Graviton.
This file does not measure Graviton or COOL, and it has no execution role, no
account id, and no image URI in ECR.
