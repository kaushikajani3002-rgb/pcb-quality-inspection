# Training Pipeline

This directory contains all offline ML training and fine-tuning scripts.
These scripts are **never** used during application runtime.

## Scripts

| Script | Purpose | Usage |
|:--|:--|:--|
| `train_component_yolo.py` | Train YOLO11m for PCB component detection | `python training/train_component_yolo.py --data datasets/Master_PCB_Dataset/data.yaml` |
| `finetune_defect_yolo.py` | Fine-tune YOLO for PCB defect/crack detection | `python training/finetune_defect_yolo.py` |

## Output

Trained weights are saved to `runs/detect/`. After training, copy the best checkpoint to `models/`:

```bash
cp runs/detect/pcb_components/weights/best.pt models/component_yolo11m.pt
```
