## Installation
  - Download program
```
git clone https://github.com/brian2lee/PHA_YOLO.git
```
  - Create by Docker
```
cd PHA_YOLO/docker
docker build -t pha .
docker run -it -d \
  --gpus all \
  --name [YourContainerName] \
  -v /path/to/workspace:/workspace/PHA_YOLO \
  --shm-size=64g \
  -w /workspace/PHA_YOLO \
  pha
cd PHA_YOLO 
docker exec -it [YourContainerName] /bin/bash
```
  - install env from souce.
```
pip install -e .
```
## Train
  - Datasets prepare

Download [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-screws), [HRSC2016](https://www.kaggle.com/datasets/guofeng/hrsc2016) and put them under <PHA-YOLO_HOME>/datasets.

  - Start Training
```
python3 tools/train/train_test.py
```
  - Resume Training
  
Change checkpoint in <PHA-YOLO_HOME>/tools/train/resume_pha.py
```
python3 tools/train/resume_pha.py
```
