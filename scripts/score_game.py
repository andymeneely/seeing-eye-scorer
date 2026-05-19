#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from seeing_eye_scorer.scoring import load_json, score_detection_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply game scoring rules to detection output.")
    parser.add_argument("detections", help="Detection JSON produced by run_inference.py.")
    parser.add_argument("config", help="Game scoring config JSON.")
    parser.add_argument("--frame", default="latest", help="Frame index to score, or latest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detection_payload = load_json(args.detections)
    config = load_json(args.config)
    scored = score_detection_payload(detection_payload, config, frame_selector=args.frame)
    print(json.dumps(scored, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
