"""Train a person/boat detector on the exported detection set.

Install the training extra before running: pip install ultralytics.
Weights are written under data/training and are not committed.
DJI_0804 is not in this set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.trained_detector import inference_device  # noqa: E402

DATASET = Path("data/training/detect/dataset.yaml")
RUNS = Path("data/training/runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--name", default="person-boat")
    parser.add_argument("--close-mosaic", type=int, default=10)
    args = parser.parse_args()
    if not DATASET.exists():
        raise SystemExit("Missing dataset. Run scripts/build_detection_set.py first.")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install ultralytics in this environment before training.") from error
    model = YOLO(args.model)
    model.train(
        data=str(DATASET),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=inference_device(),
        project=str(RUNS.resolve()),
        name=args.name,
        exist_ok=True,
        workers=2,
        close_mosaic=args.close_mosaic,
    )


if __name__ == "__main__":
    main()
