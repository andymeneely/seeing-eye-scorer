# Fine-tuning dataset workflow

This folder is organized for YOLO-style training data and for batches of new
images that still need manual annotation.

## Folder layout

```text
datasets/fine_tuning/
├── data.yaml
├── incoming/
│   └── .gitkeep
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Workflow

1. Add each new batch of raw images to `incoming/` first.
2. Review and rename images as needed before annotation.
3. Move each image into exactly one split under `images/train`, `images/val`,
   or `images/test`.
4. Annotate each image in YOLO format and save the matching `.txt` file in the
   corresponding `labels/<split>/` folder.
5. Keep image and label filenames aligned. For example,
   `images/train/frame-001.jpg` must pair with
   `labels/train/frame-001.txt`.
6. Update `data.yaml` if the class list changes.

## YOLO label format

Each line in a label file should be:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates must be normalized to the range `0.0` to `1.0`, and `class_id`
must stay consistent with the order defined in `data.yaml`.
