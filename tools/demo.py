from ultralytics import YOLO
model = YOLO('/workspace/PHA_YOLO/runs/obb/your_train_name/weights/best.pt')
results = model('your_datasets_path/hrscms_OR_hrsc_OR_dota/images/test', save=True, show_labels=False, name="demo")

