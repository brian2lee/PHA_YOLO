from ultralytics import YOLO
 
def main():
    # model = YOLO('/ultralytics/cfg/models/v8/yolov8-obb_.yaml')
    model.train(data='/ultralytics/cfg/datasets/MVTEC.yaml', epochs=100, imgsz=640, batch=64,device=[0,1,2,3] ,  amp=False, optimizer='SGD', name="v8_mvtec" )
if __name__ == '__main__':
    main()
