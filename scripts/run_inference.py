#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a webcam, image, or video source.")
    parser.add_argument("weights", help="Trained YOLO weights file.")
    parser.add_argument("--source", default="0", help="Camera index, image path, or video path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=1, help="Maximum frames to process. Use 0 for no limit.")
    parser.add_argument("--display", action="store_true", help="Show annotated frames while running.")
    parser.add_argument("--output-json", default="detections.json", help="Where to write detection output.")
    return parser.parse_args()


def is_camera_source(source: str) -> bool:
    return source.isdigit()


def serialize_result(result, frame_index: int) -> dict:
    names = result.names
    detections = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            cls_index = int(box.cls.item())
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "label": names.get(cls_index, str(cls_index)),
                    "class_id": cls_index,
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": xyxy,
                }
            )

    return {
        "frame_index": frame_index,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
    }


def run_camera_inference(model, source: str, conf: float, display: bool, max_frames: int) -> list[dict]:
    import cv2

    capture = cv2.VideoCapture(int(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera source {source}")

    frames = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, conf=conf, verbose=False)[0]
            frames.append(serialize_result(result, frame_index))

            if display:
                cv2.imshow("Seeing Eye Scorer Inference", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if max_frames and frame_index >= max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return frames


def run_file_inference(model, source: str, conf: float) -> list[dict]:
    results = model.predict(source=source, conf=conf, verbose=False)
    return [serialize_result(result, frame_index) for frame_index, result in enumerate(results)]


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required. Install dependencies from requirements.txt first.") from exc

    model = YOLO(args.weights)
    if is_camera_source(args.source):
        frames = run_camera_inference(model, args.source, args.conf, args.display, args.max_frames)
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            raise SystemExit(f"Source path does not exist: {source_path}")
        frames = run_file_inference(model, str(source_path), args.conf)

    payload = {
        "model": args.weights,
        "source": args.source,
        "frames": frames,
    }
    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote detections to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
