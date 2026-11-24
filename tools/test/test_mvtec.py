from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('/workspace/PHA_YOLO/runs/obb/pha_test_b4_600/weights/best.pt')
metrics = model.val(
    data='/workspace/PHA_YOLO/ultralytics/cfg/datasets/mvtec.yaml',
    split='test',
    task="obb",
    conf=0.01,
    nms=True,
)
