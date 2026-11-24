from ultralytics import YOLO
model = YOLO('/workspace/PHA_YOLO/runs/obb/pha_test_b4_600/weights/best.pt')
results = model('/workspace/PHA_YOLO/datasets/MVTec_screws/images/test', save=True, show_labels=False, name="demo")

