from ultralytics import YOLO

checkpoint = (
    "/workspace/PHA_YOLO/runs/obb/"
    "pha_dscov_hrsc2016_b4_600/weights/last.pt"
)

model = YOLO(checkpoint, task="obb")
model.train(resume=True)
