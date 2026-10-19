from ultralytics import YOLO
 
def main():
    model = YOLO('/ultralytics/cfg/models/v9/yolov9c_PHA-uvc.yaml', task="obb").load('/workspace/PHA_YOLO/yolov9c.pt')  # build from YAML and transfer weights
    model.train(data='/ultralytics/cfg/datasets/DOTA.yaml', epochs=100, imgsz=640, batch=4, device=[0,1,2,3], amp=False, optimizer='SGD', name="PHA-YOLO_DOTA")
if __name__ == '__main__':
    main()
