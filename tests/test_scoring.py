from seeing_eye_scorer.scoring import score_detection_payload


def test_score_detection_payload_uses_latest_frame_and_applies_rules() -> None:
    detections = {
        "frames": [
            {"frame_index": 0, "detections": [{"label": "coin"}]},
            {
                "frame_index": 1,
                "detections": [
                    {"label": "coin"},
                    {"label": "coin"},
                    {"label": "crown-card"},
                    {"label": "castle"},
                    {"label": "banner"},
                    {"label": "meeple"},
                    {"label": "meeple"},
                    {"label": "meeple"},
                    {"label": "meeple"},
                ],
            },
        ]
    }
    config = {
        "game": "prototype",
        "rules": [
            {"name": "Coins", "type": "count", "class": "coin", "points_each": 1},
            {"name": "Crown cards", "type": "count", "class": "crown-card", "points_each": 3},
            {"name": "Castle set bonus", "type": "set", "classes": {"castle": 1, "banner": 1}, "points": 5},
            {"name": "Meeple majority bonus", "type": "bonus", "class": "meeple", "at_least": 4, "points": 7},
        ],
    }

    scored = score_detection_payload(detections, config)

    assert scored["frame_index"] == 1
    assert scored["counts"] == {"coin": 2, "crown-card": 1, "castle": 1, "banner": 1, "meeple": 4}
    assert scored["total"] == 17
