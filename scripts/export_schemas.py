"""Export independent JSON Schemas for cross-team integration and generated clients."""

import argparse
import json
from pathlib import Path

from src.server.schemas import (
    AgentActionPayload,
    BoundingBox,
    LiveEnvelope,
    OverrideRequest,
    SystemTelemetry,
    VisionState,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/schemas"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for model in (
        BoundingBox,
        VisionState,
        AgentActionPayload,
        SystemTelemetry,
        LiveEnvelope,
        OverrideRequest,
    ):
        destination = args.output / f"{model.__name__}.json"
        destination.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        print(destination)


if __name__ == "__main__":
    main()
