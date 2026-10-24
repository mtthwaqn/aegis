"""Run the live detector on one frame and record the agent's action.

No hardware is commanded. Geographic coordinates stay empty unless the vision
state already carries a Member 3 fix, so a raw detection escalates instead of
inventing a drop point.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.runner import run_agent, trace_payload  # noqa: E402
from src.perception.video_processing import analyze, to_vision_state  # noqa: E402


def detect_and_act(frame: np.ndarray, frame_id: int, timestamp: datetime) -> dict[str, object]:
    result = analyze(frame)
    state = to_vision_state(result, frame_id, timestamp)
    return trace_payload(run_agent(state, frame))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--frame-id", type=int, default=1)
    args = parser.parse_args()
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
    else:
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        frame[:] = (150, 145, 80)
    print(json.dumps(detect_and_act(frame, args.frame_id, datetime.now(UTC)), indent=2))


if __name__ == "__main__":
    main()
