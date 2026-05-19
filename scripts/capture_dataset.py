#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time


IMAGE_EXT = ".jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture tabletop images from a USB webcam for YOLO labeling.")
    parser.add_argument("output_dir", help="Dataset directory to create or reuse.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--classes", nargs="*", default=[], help="Class names for YOLO data.yaml.")
    parser.add_argument("--width", type=int, help="Requested capture width.")
    parser.add_argument("--height", type=int, help="Requested capture height.")
    parser.add_argument("--interval", type=float, default=5.0, help="Auto-capture interval in seconds.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Dataset split to save into.")
    parser.add_argument("--max-images", type=int, help="Stop after this many captured images.")
    return parser.parse_args()


def write_dataset_yaml(dataset_dir: Path, classes: list[str]) -> None:
    lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  - {name}" for name in classes)
    (dataset_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dataset_dir / "classes.txt").write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")


def ensure_layout(dataset_dir: Path, classes: list[str]) -> tuple[Path, Path, Path]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "captures.jsonl"
    write_dataset_yaml(dataset_dir, classes)
    return images_dir, labels_dir, manifest_path


def save_capture(frame, images_dir: Path, labels_dir: Path, manifest_path: Path, split: str, capture_index: int) -> Path:
    import cv2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"tabletop_{timestamp}_{capture_index:04d}"
    image_path = images_dir / split / f"{stem}{IMAGE_EXT}"
    label_path = labels_dir / split / f"{stem}.txt"

    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Unable to write image to {image_path}")
    label_path.write_text("", encoding="utf-8")
    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "image": str(image_path.relative_to(images_dir.parent)),
        "label": str(label_path.relative_to(labels_dir.parent)),
        "split": split,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return image_path


def main() -> int:
    args = parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install dependencies from requirements.txt first.") from exc

    dataset_dir = Path(args.output_dir)
    images_dir, labels_dir, manifest_path = ensure_layout(dataset_dir, args.classes)

    capture = cv2.VideoCapture(args.camera_index)
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")

    auto_capture = False
    last_capture_at = 0.0
    capture_index = 0

    print("Press c to capture, a to toggle auto-capture, and q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Camera read failed.")

            overlay = frame.copy()
            status = f"split={args.split} auto={'on' if auto_capture else 'off'} saved={capture_index}"
            cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Seeing Eye Scorer Capture", overlay)

            if auto_capture and time.monotonic() - last_capture_at >= args.interval:
                image_path = save_capture(frame, images_dir, labels_dir, manifest_path, args.split, capture_index)
                capture_index += 1
                last_capture_at = time.monotonic()
                print(f"Captured {image_path}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                image_path = save_capture(frame, images_dir, labels_dir, manifest_path, args.split, capture_index)
                capture_index += 1
                last_capture_at = time.monotonic()
                print(f"Captured {image_path}")
            elif key == ord("a"):
                auto_capture = not auto_capture
                print(f"Auto-capture {'enabled' if auto_capture else 'disabled'}")
            elif key == ord("q"):
                break

            if args.max_images is not None and capture_index >= args.max_images:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
