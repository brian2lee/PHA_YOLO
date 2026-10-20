from ultralytics import YOLO

checkpoint = (
    "/workspace/PHA_YOLO/runs/obb/"
    "pha_test_b4_600/weights/last.pt"
)

model = YOLO(checkpoint, task="obb")
model.train(resume=True)
