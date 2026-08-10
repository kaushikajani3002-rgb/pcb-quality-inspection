# Dataset Policy

## Important

Datasets are **NOT** stored in this repository. They are too large for version control.

The `Dataset/` directory at the project root contains downloaded ZIP archives and unpacked datasets that are used exclusively by training scripts in `training/`.

## Current Datasets

| Dataset | Location | Size | Used By |
|:--|:--|:--|:--|
| Arduino+ESP32 Cam | `Dataset/Arduino uno - Esp32 cam.v1i.yolov11.zip` | 12 MB | `training/train_component_yolo.py` |
| Arduino Uno v2 | `Dataset/Arduino_Uno.v2i.yolov11.zip` | 54 MB | `training/train_component_yolo.py` |
| Generic PCB | `Dataset/pcb.v1i.yolov11.zip` | 651 MB | `training/train_component_yolo.py` |
| DeepPCB | `Dataset/archive/DeepPCB/` | ~91 MB | `training/finetune_defect_yolo.py` |
| DsPCBSD+ | `Dataset/archive/DsPCBSD+/` | Unpacked | `training/finetune_defect_yolo.py` |
| HRIPCB | `Dataset/archive/HRIPCB/` | Unpacked | Future |
| TDD-PCB | `Dataset/archive/TDD-PCB/` | Unpacked | `experiments/PCB_11m_1.py` |
| Merged | `Dataset/archive/merged/` | Unpacked | `training/finetune_defect_yolo.py` |

## How Training Scripts Find Datasets

Training scripts reference datasets via their `data.yaml` path:

```bash
python training/train_component_yolo.py --data Dataset/archive/TDD-PCB/data.yaml
```

## Dataset Format

All datasets follow the Ultralytics YOLO format:

```
dataset_name/
├── data.yaml       # Class names, paths to train/val/test splits
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
