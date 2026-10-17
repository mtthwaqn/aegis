"""Add SeaDronesSee swimmer and boat boxes to the YOLO detection set.

Images are resized like the live feed. Jetski, buoy, and life-saving appliances
are omitted. The SeaDronesSee test split is not in this folder. DJI_0804 stays
in the MOBDrone export only as a held-out score set.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_detection_set import fit_frame, yolo_line  # noqa: E402

SOURCE = Path("data/seadronessee")
OUT = Path("data/training/detect")
KEPT = {"swimmer": 0, "boat": 1}


def export_split(split: str) -> tuple[int, int]:
    document = json.loads((SOURCE / "annotations" / f"instances_{split}.json").read_text())
    names = {item["id"]: item["name"] for item in document["categories"]}
    boxes: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for annotation in document["annotations"]:
        if annotation.get("iscrowd"):
            continue
        class_id = KEPT.get(names[annotation["category_id"]])
        if class_id is None:
            continue
        bbox = [float(value) for value in annotation["bbox"]]
        boxes[annotation["image_id"]].append((class_id, bbox))
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    kept_boxes = 0
    for image in document["images"]:
        items = boxes.get(image["id"])
        if not items:
            continue
        source = SOURCE / "images" / split / image["file_name"]
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"missing {source}")
            continue
        fitted, ratio = fit_frame(frame)
        height, width = fitted.shape[:2]
        lines = [
            line
            for class_id, bbox in items
            if (line := yolo_line(bbox, ratio, width, height, class_id))
        ]
        if not lines:
            continue
        stem = f"sds_{Path(image['file_name']).stem}"
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), fitted, [cv2.IMWRITE_JPEG_QUALITY, 80])
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1
        kept_boxes += len(lines)
        if written % 500 == 0:
            print(f"{split} {written}", flush=True)
    return written, kept_boxes


def main() -> None:
    if not (SOURCE / "annotations" / "instances_train.json").exists():
        raise SystemExit("Missing SeaDronesSee annotations under data/seadronessee.")
    train_count, train_boxes = export_split("train")
    val_count, val_boxes = export_split("val")
    print(f"seadronessee train {train_count} boxes {train_boxes} val {val_count} boxes {val_boxes}")


if __name__ == "__main__":
    main()
