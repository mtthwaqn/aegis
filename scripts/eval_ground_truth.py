"""Score perception boxes against MOBDrone ground truth.

Annotations are the five-class COCO file in data/annotations/. That directory
is gitignored. Person maps to survivor, boat to hazard, and wood to debris.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from src.perception.video_processing import analyze

ANNOTATIONS = Path("data/annotations/annotations_5_custom_classes.json")
VIDEOS = Path("data/video_split_fullhd")
CLASS_OF = {"person": "survivor", "boat": "hazard", "wood": "debris"}
MIN_AREA = {"person": 800.0, "boat": 4500.0, "wood": 150.0}


def fit_frame(frame: cv2.Mat) -> tuple[cv2.Mat, float]:
    """Match the acquisition resize so boxes share the contract frame."""
    height, width = frame.shape[:2]
    ratio = min(1.0, 1280 / width, 720 / height)
    if ratio < 1:
        frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
    return frame, ratio


def iou(left: list[float], right: list[float]) -> float:
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def matched(predictions: list[list[float]], truths: list[list[float]], threshold: float) -> int:
    pairs: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            score = iou(prediction, truth)
            if score >= threshold:
                pairs.append((score, pred_index, truth_index))
    pairs.sort(reverse=True)
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    hits = 0
    for _, pred_index, truth_index in pairs:
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
        hits += 1
    return hits


def load_samples(prefix: str, stride: int) -> dict[str, dict[int, list[tuple[str, list[float]]]]]:
    document = json.loads(ANNOTATIONS.read_text())
    names = {item["id"]: item["name"] for item in document["categories"]}
    grouped: dict[str, dict[int, list[tuple[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    images: dict[int, tuple[str, int]] = {}
    for image in document["images"]:
        video, index = Path(image["file_name"]).stem.rsplit("_", 1)
        if not video.startswith(prefix):
            continue
        frame_index = int(index)
        if frame_index % stride != 0:
            continue
        images[image["id"]] = (video, frame_index)
        grouped[video][frame_index] = []
    for annotation in document["annotations"]:
        located = images.get(annotation["image_id"])
        if located is None or annotation.get("iscrowd"):
            continue
        video, frame_index = located
        name = names[annotation["category_id"]]
        grouped[video][frame_index].append((name, [float(value) for value in annotation["bbox"]]))
    return grouped


def score(prefix: str, stride: int) -> None:
    samples = load_samples(prefix, stride)
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
            result = analyze(fitted)
            frames += 1
            by_class: dict[str, list[list[float]]] = defaultdict(list)
            areas: dict[str, list[float]] = defaultdict(list)
            for name, bbox in truths:
                if name not in CLASS_OF:
                    continue
                scaled = [value * ratio for value in bbox]
                by_class[name].append(scaled)
                areas[name].append(scaled[2] * scaled[3])
            predictions = {
                classification: [
                    list(detection.coordinates)
                    for detection in result.detections
                    if detection.classification == classification
                ]
                for classification in CLASS_OF.values()
            }
            for name, classification in CLASS_OF.items():
                found = predictions[classification]
                expected = by_class[name]
                large = [
                    box
                    for box, area in zip(expected, areas[name], strict=True)
                    if area >= MIN_AREA[name] * ratio * ratio
                ]
                for threshold, bucket in totals[name].items():
                    hits = matched(found, expected, threshold)
                    large_hits = matched(found, large, threshold)
                    bucket["tp"] += hits
                    bucket["fp"] += len(found) - hits
                    bucket["fn"] += len(expected) - hits
                    bucket["large_tp"] += large_hits
                    bucket["large_fn"] += len(large) - large_hits
        capture.release()
        print(f"scored {video} ({frames} frames)")
    print(f"frames {frames} stride {stride} prefix {prefix}")
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


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="DJI_0804")
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()
    score(args.prefix, args.stride)


if __name__ == "__main__":
    main()
