from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('/workspace/PHA_YOLO/runs/obb/pha_hrsc2016_b4_600/weights/best.pt')
metrics = model.val(
    data='/workspace/PHA_YOLO/ultralytics/cfg/datasets/hrsc2016.yaml',
    split='test',
    task="obb",
    conf=0.01,
    nms=True,
)
