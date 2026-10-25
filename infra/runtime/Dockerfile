# Local image definition. Do not push this to ECR and do not register it with ECS.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts

RUN pip install --no-cache-dir -r requirements.lock.txt \
    && pip install --no-cache-dir --no-deps .

COPY infra/runtime/start.sh /app/start.sh
COPY infra/runtime/media/clip.mp4 /app/media/clip.mp4
RUN chmod +x /app/start.sh

ENV AEGIS_DETECTOR=classical \
    AEGIS_VIDEO_SOURCE=/app/media/clip.mp4 \
    AEGIS_DEPLOYMENT_VERIFIED=false \
    AEGIS_MOCK_ACTIONS=false \
    AEGIS_COMPUTE_INSTANCE="Fargate X86_64, 1 vCPU, 2 GB, ap-southeast-2" \
    AEGIS_KINESIS_REGION=ap-southeast-2

EXPOSE 8000

CMD ["/app/start.sh"]
