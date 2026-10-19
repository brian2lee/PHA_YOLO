from ultralytics import YOLO
 
def main():
    # model = YOLO('/ultralytics/cfg/models/cspdarknet/.yaml')
    model.train(data='/ultralytics/cfg/datasets/MVTEC.yaml', epochs=100, imgsz=640,  amp=False, optimizer='SGD', batch=64,device=[0,1,2,3] , name="x_mvtec" )
if __name__ == '__main__':
    main()
