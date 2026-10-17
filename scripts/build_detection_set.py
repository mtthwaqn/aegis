"""Export a small YOLO set from the local MOBDrone videos.

The official test clips (DJI_0804) are left out. Images are resized the same way
the live feed is, so boxes match the vision contract. Output stays under data/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from src.perception.trained_classes import TRAIN_CLASSES

ANNOTATIONS = Path("data/annotations/annotations_5_custom_classes.json")
VIDEOS = Path("data/video_split_fullhd")
OUT = Path("data/training/detect")
VAL_PREFIXES = ("DJI_0915_0001", "DJI_0915_0002", "DJI_0915_0003")
CLASS_IDS = {name: index for index, name in enumerate(TRAIN_CLASSES)}


def fit_frame(frame: cv2.Mat) -> tuple[cv2.Mat, float]:
    height, width = frame.shape[:2]
    ratio = min(1.0, 1280 / width, 720 / height)
    if ratio < 1:
        frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
    return frame, ratio


def split_of(video: str) -> str | None:
    if video.startswith("DJI_0804"):
        return None
    if video.startswith(VAL_PREFIXES):
        return "val"
    return "train"


def yolo_line(
    bbox: list[float],
    ratio: float,
    width: int,
    height: int,
    class_id: int,
) -> str | None:
    x, y, w, h = (value * ratio for value in bbox)
    x = min(max(x, 0.0), width - 1)
    y = min(max(y, 0.0), height - 1)
    w = min(w, width - x)
    h = min(h, height - y)
    if w < 2 or h < 2:
        return None
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def load_frames(stride: int) -> dict[str, dict[str, dict[int, list[tuple[int, list[float]]]]]]:
    document = json.loads(ANNOTATIONS.read_text())
    names = {item["id"]: item["name"] for item in document["categories"]}
    grouped: dict[str, dict[str, dict[int, list[tuple[int, list[float]]]]]] = {
        "train": {},
        "val": {},
    }
    images: dict[int, tuple[str, str, int]] = {}
    for image in document["images"]:
        video, index_text = Path(image["file_name"]).stem.rsplit("_", 1)
        split = split_of(video)
        frame_index = int(index_text)
        if split is None or frame_index % stride:
            continue
        images[image["id"]] = (split, video, frame_index)
        grouped[split].setdefault(video, {}).setdefault(frame_index, [])
    for annotation in document["annotations"]:
        located = images.get(annotation["image_id"])
        if located is None or annotation.get("iscrowd"):
            continue
        name = names[annotation["category_id"]]
        if name not in CLASS_IDS:
            continue
        split, video, frame_index = located
        bbox = [float(value) for value in annotation["bbox"]]
        grouped[split][video][frame_index].append((CLASS_IDS[name], bbox))
    return grouped


def export_split(
    split: str,
    frames: dict[str, dict[int, list[tuple[int, list[float]]]]],
    limit: int,
    empty_limit: int,
    seed: int,
) -> int:
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    chosen: list[tuple[str, int, list[tuple[int, list[float]]]]] = []
    empties: list[tuple[str, int]] = []
    for video, by_index in sorted(frames.items()):
        for frame_index, boxes in sorted(by_index.items()):
            if boxes:
                chosen.append((video, frame_index, boxes))
            else:
                empties.append((video, frame_index))
    rng = random.Random(seed)
    rng.shuffle(chosen)
    rng.shuffle(empties)
    selected = chosen[:limit]
    selected.extend((video, index, []) for video, index in empties[:empty_limit])
    written = 0
    by_video: dict[str, list[tuple[int, list[tuple[int, list[float]]]]]] = {}
    for video, frame_index, boxes in selected:
        by_video.setdefault(video, []).append((frame_index, boxes))
    for video, items in sorted(by_video.items()):
        path = VIDEOS / f"{video}.mp4"
        if not path.exists():
            print(f"missing {path.name}")
            continue
        capture = cv2.VideoCapture(str(path))
        for frame_index, boxes in sorted(items):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            fitted, ratio = fit_frame(frame)
            height, width = fitted.shape[:2]
            lines = [
                line
                for class_id, bbox in boxes
                if (line := yolo_line(bbox, ratio, width, height, class_id))
            ]
            stem = f"{video}_{frame_index:06d}"
            cv2.imwrite(str(image_dir / f"{stem}.jpg"), fitted, [cv2.IMWRITE_JPEG_QUALITY, 80])
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            written += 1
        capture.release()
        print(f"{split} {video} ({written})")
    return written


def write_yaml() -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(TRAIN_CLASSES))
    text = f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    (OUT / "dataset.yaml").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--train-limit", type=int, default=400)
    parser.add_argument("--val-limit", type=int, default=80)
    parser.add_argument("--empty-train", type=int, default=40)
    parser.add_argument("--empty-val", type=int, default=8)
    args = parser.parse_args()
    grouped = load_frames(args.stride)
    train_count = export_split("train", grouped["train"], args.train_limit, args.empty_train, 2)
    val_count = export_split("val", grouped["val"], args.val_limit, args.empty_val, 3)
    write_yaml()
    print(f"wrote train {train_count} val {val_count}")


if __name__ == "__main__":
    main()
