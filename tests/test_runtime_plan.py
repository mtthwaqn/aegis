import json
from pathlib import Path

import numpy as np
import pytest

from src.perception.local_benchmark import measure_classical

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "infra" / "runtime"


def test_runtime_image_is_not_a_deployment() -> None:
    dockerfile = (RUNTIME / "Dockerfile").read_text(encoding="utf-8")
    task = json.loads((RUNTIME / "ecs-task-definition.json").read_text(encoding="utf-8"))
    notes = (RUNTIME / "README.md").read_text(encoding="utf-8")
    assert "AEGIS_DETECTOR=classical" in dockerfile
    assert "AEGIS_DEPLOYMENT_VERIFIED=false" in dockerfile
    assert "amazonaws.com" not in dockerfile
    assert "AKIA" not in dockerfile
    assert task["containerDefinitions"][0]["image"] == "aegis-command-center:local"
    assert task["runtimePlatform"]["cpuArchitecture"] == "ARM64"
    environment = {item["name"]: item["value"] for item in task["containerDefinitions"][0]["environment"]}
    assert environment["AEGIS_DEPLOYMENT_VERIFIED"] == "false"
    assert environment["AEGIS_DETECTOR"] == "classical"
    assert environment["AEGIS_VIDEO_SOURCE"] == "mock"
    assert "do not" in notes.lower()
    blob = dockerfile + json.dumps(task) + notes
    for secret in ("AWS_SECRET_ACCESS_KEY", "AKIA", "aws_access_key_id"):
        assert secret not in blob


def test_kinesis_plan_still_excludes_the_application() -> None:
    template = (ROOT / "infra" / "aws" / "kinesis-stream.yaml").read_text(encoding="utf-8")
    for denied in ("AWS::ECS::", "AWS::ECR::", "AWS::EC2::"):
        assert denied not in template


def test_local_benchmark_does_not_claim_graviton_or_cool() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:] = (150, 145, 80)
    report = measure_classical(frame, repeats=1)
    assert report["detector"] == "classical"
    assert report["locked"] is True
    assert report["graviton_measured"] is False
    assert report["cool_measured"] is False
    assert report["deployment_verified"] is False
    assert report["p95_ms"] >= 0


def test_benchmark_refuses_to_time_the_trained_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_DETECTOR", "trained")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="locked to classical"):
        measure_classical(frame, repeats=1)
