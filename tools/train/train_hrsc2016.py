from ultralytics import YOLO
 
def main():
    model = YOLO('/ultralytics/cfg/models/v9/yolov9c_PHA.yaml', task="obb").load('yolov9c.pt')  # build from YAML and transfer weights
    model.train(data='/ultralytics/cfg/datasets/hrsc.yaml', epochs=100, imgsz=640, batch=16, device=[0,1,2,3], amp=False, optimizer='SGD', name="PHA-YOLO_hrsc2016")
if __name__ == '__main__':
    main()
    
