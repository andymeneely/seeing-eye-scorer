# seeing-eye-scorer

Seeing Eye Scorer is a laptop-controlled computer vision system for board game
scoring. The goal is to mount a USB webcam above a gaming table, recognize game
components with YOLO, and total their score automatically.

## Requirements

This repository is intended to support a workflow with:

- an overhead USB webcam aimed at the play area
- a laptop that captures the camera feed and runs inference locally
- YOLO-based object detection for cards, tokens, meeples, tiles, or other
  scoring components
- fine-tuning on your own table layout, lighting, and game pieces
- per-game scoring rules that convert detections into totals

## Proposed system design

### 1. Capture

- Connect a standard USB webcam to a laptop
- Continuously capture frames of the tabletop from an overhead mount
- Keep the camera position stable so the model sees a consistent viewpoint

### 2. Detection

- Start from a YOLO model that can be fine-tuned for custom classes
- Train classes for the components that matter to scoring in each game
- Run inference locally on the laptop so the setup works offline at the table

### 3. Fine-tuning workflow

- Record sample images from the actual table and webcam you plan to use
- Label the board game components visible in those images
- Fine-tune the YOLO model on that labeled dataset
- Export the trained weights for local scoring runs

### 4. Scoring

- Convert detected components into game-specific score events
- Keep scoring rules separate from model training so support for new games is a
  configuration problem instead of a model rewrite
- Sum the detected values into a live score total for the current game

## Minimum viable workflow

1. Mount the USB webcam above the table
2. Connect it to a laptop running the scorer
3. Capture and label images for one game
4. Fine-tune a YOLO model on those images
5. Run local detection and map detections to that game's scoring rules

## Prototype scripts

The repository now includes a minimal Python prototype for the workflow above.
Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 1. Capture webcam images into a YOLO dataset

```bash
python scripts/capture_dataset.py data/my-game --classes coin crown-card castle banner meeple
```

- opens the USB webcam with a live preview
- press `c` to capture a frame
- press `a` to toggle timed auto-capture
- press `q` to quit
- writes images, empty YOLO label files, `captures.jsonl`, `classes.txt`, and `data.yaml`

Label the generated images with any YOLO-compatible labeling tool by filling in
those `.txt` files.

### 2. Fine-tune a YOLO model locally

```bash
python scripts/train_yolo.py data/my-game/data.yaml --model yolo11n.pt --epochs 50
```

This uses Ultralytics YOLO for training and prints the path to the best weights.
Pass `--export-format onnx` if you also want an exported runtime artifact.

### 3. Run local inference on a webcam or file

```bash
python scripts/run_inference.py runs/seeing-eye-scorer/prototype/weights/best.pt --source 0 --display --max-frames 25
```

This writes detection output to `detections.json` by default. Use an image or
video path instead of `0` to score saved media.

### 4. Apply per-game scoring rules

```bash
python scripts/score_game.py examples/sample_detections.json config/games/prototype_game.json
```

The sample config shows three prototype rule types:

- `count`: points per detected class
- `set`: bonus for completing a required combination of classes
- `bonus`: fixed bonus when a class count reaches a threshold

## Files

- `scripts/capture_dataset.py`: webcam capture and dataset bootstrap
- `scripts/train_yolo.py`: Ultralytics fine-tuning entry point
- `scripts/run_inference.py`: detection runner for webcam, images, or video
- `scripts/score_game.py`: game scoring entry point
- `config/games/prototype_game.json`: example score rules
- `examples/sample_detections.json`: example detection payload

## Initial implementation priorities

1. webcam capture from USB
2. YOLO inference on laptop
3. dataset capture and labeling flow
4. fine-tuning pipeline for custom games
5. configurable score calculation per game
