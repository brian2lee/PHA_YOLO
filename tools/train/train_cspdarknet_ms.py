from ultralytics import YOLO
 
def main():
    # model = YOLO('/ultralytics/cfg/models/cspdarknet/.yaml')
    model.train(data='/ultralytics/cfg/datasets/HRSC2016_MS.yaml', epochs=100, imgsz=640, batch=64,device=[0,1,2,3] , amp=False, optimizer='SGD', name="x_ms" )
if __name__ == '__main__':
    main()
