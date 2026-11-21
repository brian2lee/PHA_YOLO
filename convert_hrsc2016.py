from pathlib import Path
import shutil
import math
import xml.etree.ElementTree as ET
import cv2


# Dataset root
ROOT = Path("/workspace/PHA_YOLO/datasets/HRSC2016")

IMAGE_DIR = ROOT / "AllImages"
ANNOTATION_DIR = ROOT / "Annotations"
IMAGESETS_DIR = ROOT / "ImageSets"

# Converted dataset directories
OUTPUT_IMAGE_DIR = ROOT / "images"
OUTPUT_LABEL_DIR = ROOT / "labels"

SPLITS = ["train", "val", "test"]


def get_value(parent, tag):
    element = parent.find(tag)

    if element is None or element.text is None:
        raise ValueError(f"Missing XML field: {tag}")

    return element.text.strip()


def rotated_box_to_points(cx, cy, w, h, angle):
    """
    Convert HRSC2016 robndbox to four corner points.
    Angle is in radians.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    half_w = w / 2
    half_h = h / 2

    corners = [
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
    ]

    points = []

    for x, y in corners:
        px = cx + x * cos_a - y * sin_a
        py = cy + x * sin_a + y * cos_a
        points.append((px, py))

    return points


def convert_annotation(xml_path, image_width, image_height):
    root = ET.parse(xml_path).getroot()
    labels = []

    for obj in root.findall("object"):
        robndbox = obj.find("robndbox")

        if robndbox is None:
            raise ValueError(f"No robndbox found in {xml_path}")

        cx = float(get_value(robndbox, "cx"))
        cy = float(get_value(robndbox, "cy"))
        w = float(get_value(robndbox, "w"))
        h = float(get_value(robndbox, "h"))
        angle = float(get_value(robndbox, "angle"))

        points = rotated_box_to_points(cx, cy, w, h, angle)

        normalized = []

        for x, y in points:
            x = min(max(x / image_width, 0.0), 1.0)
            y = min(max(y / image_height, 0.0), 1.0)
            normalized.extend([x, y])

        # Single class: ship = 0
        labels.append(
            "0 " + " ".join(f"{v:.6f}" for v in normalized)
        )

    return labels


def main():
    for split in SPLITS:
        print(f"\nProcessing {split}...")

        image_output = OUTPUT_IMAGE_DIR / split
        label_output = OUTPUT_LABEL_DIR / split

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        split_file = IMAGESETS_DIR / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(split_file)

        image_ids = [
            line.strip().split()[0]
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

        for index, image_id in enumerate(image_ids, 1):
            image_path = IMAGE_DIR / f"{image_id}.bmp"
            xml_path = ANNOTATION_DIR / f"{image_id}.xml"

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            if not xml_path.exists():
                raise FileNotFoundError(xml_path)

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            height, width = image.shape[:2]

            labels = convert_annotation(xml_path, width, height)

            shutil.copy2(
                image_path,
                image_output / image_path.name,
            )

            label_path = label_output / f"{image_id}.txt"
            label_path.write_text(
                "\n".join(labels) + ("\n" if labels else "")
            )

            if index % 100 == 0:
                print(f"{split}: {index}/{len(image_ids)}")

        print(f"Finished {split}: {len(image_ids)} images")

    print("\nConversion completed!")
    print(f"Dataset: {ROOT}")


if __name__ == "__main__":
    main()

