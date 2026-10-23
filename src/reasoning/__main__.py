from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.server.schemas import VisionState

from .dataset_metadata import enrich_from_seadronessee, load_image_record
from .vision_state import to_reasoning_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process Member 2 VisionState with Member 3 reasoning"
    )
    parser.add_argument("vision", type=Path)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--horizontal-fov-degrees", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.annotations is None) != (args.image_id is None):
        parser.error(
            "--annotations and --image-id must be supplied together for the same source image"
        )
    if args.horizontal_fov_degrees is not None and args.annotations is None:
        parser.error("--horizontal-fov-degrees requires --annotations and --image-id")
    try:
        vision = VisionState.model_validate_json(args.vision.read_text(encoding="utf-8-sig"))
        report = {"status": "not_requested", "reason": "No matching source-image metadata supplied"}
        if args.annotations is not None:
            record = load_image_record(args.annotations, args.image_id)
            vision, report = enrich_from_seadronessee(vision, record, args.horizontal_fov_degrees)
        result = {
            "vision": vision.model_dump(mode="json"),
            "reasoning": to_reasoning_payload(vision, args.threshold),
            "georeferencing": report,
        }
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
