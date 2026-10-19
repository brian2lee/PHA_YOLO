from ultralytics import YOLO


def main():
    model = YOLO(
        "/workspace/PHA_YOLO/ultralytics/cfg/models/v8/yolov8-obb_PHA.yaml",
        task="obb",
    )

    model.train(
        data="/workspace/PHA_YOLO/ultralytics/cfg/datasets/mvtec.yaml",
        epochs=600,
        imgsz=640,
        batch=4,
        device=0,
        amp=False,
        optimizer="SGD",
        pretrained=False,
        patience=100,
        name="pha_test_b4_600",
        project="/workspace/PHA_YOLO/runs/obb",
        seed=0,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
