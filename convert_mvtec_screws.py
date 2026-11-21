import json
import math
import os
from pathlib import Path

root = Path("/workspace/PHA_YOLO/datasets/MVTec_screws")
image_root = root / "images"

splits = {
    "train": root / "mvtec_screws_train.json",
    "val": root / "mvtec_screws_val.json",
    "test": root / "mvtec_screws_test.json",
}

# Map the original COCO category IDs (1..13) to YOLO IDs (0..12).
with open(splits["train"], "r") as f:
    train_data = json.load(f)

categories = sorted(train_data["categories"], key=lambda x: x["id"])
category_map = {c["id"]: i for i, c in enumerate(categories)}

print("Classes:")
for c in categories:
    print(f'  {category_map[c["id"]]}: {c["name"]}')

for split, json_path in splits.items():
    with open(json_path, "r") as f:
        data = json.load(f)

    image_by_id = {im["id"]: im for im in data["images"]}
    annotations_by_image = {}

    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    written = 0
    clipped_boxes = 0
    skipped_boxes = 0

    for image_id, im in image_by_id.items():
        filename = im["file_name"]
        source_image = image_root / filename

        if not source_image.is_file():
            raise FileNotFoundError(f"Missing image: {source_image}")

        # Link the existing image into its split directory.
        target_image = image_dir / filename
        if target_image.is_symlink() or target_image.exists():
            target_image.unlink()
        target_image.symlink_to(source_image.resolve())
        linked += 1

        W, H = im["width"], im["height"]
        label_path = label_dir / f"{Path(filename).stem}.txt"
        lines = []

        for ann in annotations_by_image.get(image_id, []):
            # MVTec COCO-style bbox: [row, col, width, height, phi]
            row, col, bw, bh, phi = ann["bbox"]

            if bw <= 0 or bh <= 0:
                skipped_boxes += 1
                continue

            cx, cy = col, row
            c, s = math.cos(phi), math.sin(phi)

            # Four corners in consistent order.
            corners = []
            for u, v in [
                (-bw / 2, -bh / 2),
                ( bw / 2, -bh / 2),
                ( bw / 2,  bh / 2),
                (-bw / 2,  bh / 2),
            ]:
                x = cx + u * c + v * s
                y = cy - u * s + v * c
                corners.append((x, y))

            # Ultralytics requires normalized coordinates in [0, 1].
            # Clamp corners for objects whose rotated boxes cross image edges.
            if any(x < 0 or x > W or y < 0 or y > H
                   for x, y in corners):
                clipped_boxes += 1

            points = []
            for x, y in corners:
                x = min(max(x, 0.0), float(W))
                y = min(max(y, 0.0), float(H))
                points.extend([x / W, y / H])

            if len(set((round(points[i], 8), round(points[i + 1], 8))
                       for i in range(0, 8, 2))) < 3:
                skipped_boxes += 1
                continue

            cls = category_map[ann["category_id"]]
            lines.append(str(cls) + " " + " ".join(f"{p:.8f}" for p in points))
            written += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(
        f"{split}: images={linked}, boxes_written={written}, "
        f"edge_boxes_clamped={clipped_boxes}, boxes_skipped={skipped_boxes}"
    )

print(f"\nConversion complete: {root}")
