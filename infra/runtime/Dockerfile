# Local image definition. Do not push this to ECR and do not register it with ECS.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard

RUN pip install --no-cache-dir -r requirements.lock.txt \
    && pip install --no-cache-dir --no-deps .

ENV AEGIS_DETECTOR=classical \
    AEGIS_VIDEO_SOURCE=mock \
    AEGIS_DEPLOYMENT_VERIFIED=false \
    AEGIS_MOCK_ACTIONS=false \
    AEGIS_KINESIS_REGION=ap-southeast-2

EXPOSE 8000

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
