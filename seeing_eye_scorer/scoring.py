from __future__ import annotations

from collections import Counter
from typing import Any
import json
from pathlib import Path


JsonDict = dict[str, Any]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_frame(payload: Any, frame_selector: str = "latest") -> JsonDict:
    if isinstance(payload, list):
        detections = payload
        return {"frame_index": 0, "detections": detections}

    if not isinstance(payload, dict):
        raise ValueError("Detection payload must be a list or object.")

    frames = payload.get("frames")
    if frames is None:
        detections = payload.get("detections", [])
        return {"frame_index": payload.get("frame_index", 0), "detections": detections}

    if not frames:
        return {"frame_index": 0, "detections": []}

    if frame_selector == "latest":
        return frames[-1]

    frame_index = int(frame_selector)
    for frame in frames:
        if int(frame.get("frame_index", -1)) == frame_index:
            return frame

    raise ValueError(f"Frame {frame_index} not found in payload.")


def count_classes(detections: list[JsonDict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for detection in detections:
        label = str(detection.get("label") or detection.get("class_name") or detection.get("class", "unknown"))
        counts[label] += 1
    return counts


def score_detections(detections: list[JsonDict], config: JsonDict) -> JsonDict:
    counts = count_classes(detections)
    breakdown = []
    total = 0

    for rule in config.get("rules", []):
        rule_type = rule.get("type")
        rule_name = rule.get("name", rule_type or "rule")

        if rule_type == "count":
            label = rule["class"]
            units = counts[label]
            points = units * int(rule.get("points_each", 0))
        elif rule_type == "set":
            requirements = {label: int(amount) for label, amount in rule.get("classes", {}).items()}
            units = min((counts[label] // amount for label, amount in requirements.items()), default=0)
            points = units * int(rule.get("points", 0))
        elif rule_type == "bonus":
            label = rule["class"]
            units = counts[label]
            threshold = int(rule.get("at_least", 0))
            points = int(rule.get("points", 0)) if units >= threshold else 0
        else:
            raise ValueError(f"Unsupported scoring rule type: {rule_type}")

        total += points
        breakdown.append(
            {
                "name": rule_name,
                "type": rule_type,
                "units": units,
                "points": points,
            }
        )

    return {
        "game": config.get("game", "unknown"),
        "counts": dict(counts),
        "breakdown": breakdown,
        "total": total,
    }


def score_detection_payload(payload: Any, config: JsonDict, frame_selector: str = "latest") -> JsonDict:
    frame = extract_frame(payload, frame_selector=frame_selector)
    scored = score_detections(frame.get("detections", []), config)
    scored["frame_index"] = frame.get("frame_index", 0)
    return scored
