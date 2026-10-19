# PHA_YOLO
This repo is an implementation of the PHA-YOLO detector published in the following article:

Chi-Yi Tsai and Huan-Wei Hsu, "Enhancing Rotated Object Detection in Remote Sensing with a Parallel Hybrid Attention Mechanism," IEEE Access, Vol. 14, pp. 77756-77775, 2026, DOI: 10.1109/ACCESS.2026.3694595.

Please cite this article if you use our work in your research.

## Installation
  - Download program
```
git clone https://github.com/RVL224/PHA_YOLO2.git
Change the folder name to PHA_YOLO
```
  - Create by Docker
```
cd PHA_YOLO/docker
docker build -t pha .
docker run -it -d --gpus all --name [YourContainerName] -v /path/to/dataset:/workspace/PHA_YOLO/datasets -v /path/to/program:/workspace/PHA_YOLO --shm-size=64g pha
docker exec -it [YourContainerName] /bin/bash
```
  - install env from souce.
```
pip install -e .
```
## Train
  - Datasets prepare

Download [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-screws), [HRSC2016](https://www.kaggle.com/datasets/guofeng/hrsc2016), [HRSC2016-MS](https://github.com/wmchen/HRSC2016-MS), [DOTA-v1.0](https://captain-whu.github.io/DOTA/) and put them under <PHA-YOLO_HOME>/datasets.

  - Start Training
```
python3 tools/train/train_Dota.py
python3 tools/train/train_hrsc2016.py
python3 tools/train/train_hrsc2016ms.py
python3 tools/train/train_cspdarknet_ms.py
python3 tools/train/train_cspdarknet_mvtec.py
python3 tools/train/train_v8_ms.py
python3 tools/train/train_v8_mvtec.py
```
  - The trained weight file will appear in /runs/obb
## test and Demo
  - Modify the path and execute the program
```
python3 tools/demo.py
python3 tools/tests/test_dota.py
python3 tools/tests/test_hrsc_map07.py
python3 tools/tests/test_ms_map12.py
python3 tools/tests/test_mvtec.py
```
