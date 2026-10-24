"""Score trained person/boat boxes with the same IoU protocol as eval_ground_truth.

Weights stay on disk. DJI_0804 was held out of training.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ground_truth import (  # noqa: E402
    CLASS_OF,
    MIN_AREA,
    VIDEOS,
    _rate,
    fit_frame,
    load_samples,
    matched,
)

from src.perception.trained_detector import DEFAULT_WEIGHTS, inference_device  # noqa: E402

WEIGHTS = DEFAULT_WEIGHTS
LABEL_OF = {"person": "survivor", "boat": "hazard"}


def score(prefix: str, stride: int, confidence: float, weights: Path) -> None:
    from ultralytics import YOLO

    samples = load_samples(prefix, stride)
    model = YOLO(str(weights))
    totals: dict[str, dict[float, dict[str, int]]] = {
        name: {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "large_tp": 0, "large_fn": 0}
            for threshold in (0.3, 0.5)
        }
        for name in CLASS_OF
    }
    frames = 0
    for video, frames_by_index in sorted(samples.items()):
        path = VIDEOS / f"{video}.mp4"
        if not path.exists():
            print(f"missing {path}")
            continue
        capture = cv2.VideoCapture(str(path))
        for frame_index, truths in sorted(frames_by_index.items()):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            fitted, ratio = fit_frame(frame)
            frames += 1
            predicted = model.predict(
                fitted, conf=confidence, imgsz=640, verbose=False, device=inference_device()
            )
            found: dict[str, list[list[float]]] = defaultdict(list)
            for box in predicted[0].boxes:
                name = predicted[0].names[int(box.cls)]
                classification = LABEL_OF.get(name)
                if classification is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                found[classification].append([x1, y1, x2 - x1, y2 - y1])
            by_class: dict[str, list[list[float]]] = defaultdict(list)
            areas: dict[str, list[float]] = defaultdict(list)
            for name, bbox in truths:
                if name not in CLASS_OF:
                    continue
                scaled = [value * ratio for value in bbox]
                by_class[name].append(scaled)
                areas[name].append(scaled[2] * scaled[3])
            for name, classification in CLASS_OF.items():
                expected = by_class[name]
                large = [
                    box
                    for box, area in zip(expected, areas[name], strict=True)
                    if area >= MIN_AREA[name] * ratio * ratio
                ]
                boxes = found[classification]
                for threshold, bucket in totals[name].items():
                    hits = matched(boxes, expected, threshold)
                    large_hits = matched(boxes, large, threshold)
                    bucket["tp"] += hits
                    bucket["fp"] += len(boxes) - hits
                    bucket["fn"] += len(expected) - hits
                    bucket["large_tp"] += large_hits
                    bucket["large_fn"] += len(large) - large_hits
        capture.release()
        print(f"scored {video} ({frames} frames)", flush=True)
    print(f"frames {frames} stride {stride} prefix {prefix} conf {confidence}")
    for name, classification in CLASS_OF.items():
        print(f"{name} -> {classification}")
        for threshold, bucket in totals[name].items():
            precision = _rate(bucket["tp"], bucket["tp"] + bucket["fp"])
            recall = _rate(bucket["tp"], bucket["tp"] + bucket["fn"])
            large_recall = _rate(bucket["large_tp"], bucket["large_tp"] + bucket["large_fn"])
            print(
                f"  IoU {threshold:.2f}  precision {precision:.3f}  "
                f"recall {recall:.3f}  recall within size band {large_recall:.3f}  "
                f"tp {bucket['tp']} fp {bucket['fp']} fn {bucket['fn']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="DJI_0804")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Missing weights at {args.weights}")
    score(args.prefix, args.stride, args.conf, args.weights)


if __name__ == "__main__":
    main()
