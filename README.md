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
  -p 8000:8888 --expose=8888 \
  --gpus all \
  --name [YourContainerName] \
  -v /path/to/workspace:/workspace/PHA_YOLO \
  --shm-size=64g \
  -w /workspace/PHA_YOLO \
  pha \
  python3 -m jupyterlab --allow-root --ip 0.0.0.0
docker exec -it [YourContainerName] /bin/bash
```
  - install env from souce.
```
pip install -e .
```
## Train
  - Datasets prepare

Download [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-screws), [HRSC2016](https://www.kaggle.com/datasets/guofeng/hrsc2016) and put them under <PHA-YOLO_HOME>/datasets.
change folder name to HRSC2016, MVTec_screws
  - Convert datasets
change dataset location in convert_mvtec_screws.py, convert_hrsc2016.py
```
python3 convert_dataset/convert_mvtec_screws.py
python3 convert_dataset/convert_hrsc2016.py
```
  - Start Training
```
python3 tools/train/train_mvtec.py
python3 tools/train/train_HRSC2016.py
```
  - Resume Training
  
Change checkpoint in <PHA-YOLO_HOME>/tools/train/resume_pha.py
```
python3 tools/train/resume_pha.py
```
## Test
```
python3 tools/test/test_mvtec.py
python3 tools/test/test_hrsc.py
```