#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model on a captured tabletop dataset.")
    parser.add_argument("dataset", help="Path to a YOLO data.yaml file.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--project", default="runs/seeing-eye-scorer", help="Ultralytics project directory.")
    parser.add_argument("--name", default="prototype", help="Training run name.")
    parser.add_argument("--device", default=None, help="Training device, for example cpu or 0.")
    parser.add_argument("--export-format", default=None, help="Optional export format such as onnx.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required. Install dependencies from requirements.txt first.") from exc

    model = YOLO(args.model)
    results = model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")

    if args.export_format:
        exported = YOLO(best_weights).export(format=args.export_format)
        print(f"Exported model: {exported}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
