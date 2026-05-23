#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model on a captured tabletop dataset.")
    parser.add_argument("dataset", help="Path to a YOLO data.yaml file.")
    parser.add_argument("--model", default="yolo26n-obb.pt", help="Base YOLO checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--project", default="models", help="Project output directory.")
    parser.add_argument("--name", default="prototype", help="Training run name.")
    parser.add_argument("--device", default="mps", help="Training device, for example cpu or 0 or mps.")
    parser.add_argument("--export-format", default=None, help="Optional export format such as onnx.")

    parser.add_argument("--degrees", type=int, default=120, help="Data augmentation degrees to rotate images during training.")
    parser.add_argument("--translates", type=float, default=0.5, help="Data augmentation translates to shift images during training.")
    parser.add_argument("--scales", type=float, default=0.5, help="Data augmentation scales to zoom images during training.")
    parser.add_argument("--shears", type=float, default=0.2, help="Data augmentation shears to distort images during training.")
    parser.add_argument("--perspective", type=float, default=0.0, help="Data augmentation perspective to apply perspective transforms to images during training.")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Data augmentation horizontal flip probability to flip images during training.")

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
        degrees=90
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")

    if args.export_format:
        exported = YOLO(best_weights).export(format=args.export_format)
        print(f"Exported model: {exported}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
