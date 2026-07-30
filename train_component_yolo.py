"""
YOLO11m Training Script -- PCB_AOI Component Detection Model
================================================================
Fine-tunes YOLO11m (COCO-pretrained) to detect individual populated
components on a PCB assembly (IC, connector, LED, crystal, header strips,
USB port, barrel jack, voltage regulator, reset button, etc.) -- i.e. the
classes referenced by your templates/*.json files (arduino_uno.json,
esp32_devkit.json, stm32_blue_pill.json) and consumed by
inspection/inspection_engine.py, component_counter.py, missing_checker.py,
extra_checker.py, and position_checker.py.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11m on PCB_AOI component detection dataset")
    p.add_argument("--data", type=str, required=False,
                   default=r"d:\PCB\archive (1)\TDD-PCB\data.yaml",
                   help="Path to data.yaml describing train/val/test splits and component classes")
    p.add_argument("--model", type=str, default="yolo11m.pt",
                   help="Starting weights. COCO-pretrained (yolo11m.pt) fine-tuning is strongly recommended")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Image resolution to resize training samples")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (lower to 4 if facing GPU OOM)")
    p.add_argument("--device", type=str, default="-1",
                   help="'0' for first GPU, 'cpu' for CPU, '-1' for auto-selection")
    p.add_argument("--project", type=str, default="DeepPCB_Output")
    p.add_argument("--name", type=str, default="pcb_components")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a last.pt checkpoint to resume")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cls-weight", type=float, default=1.0)
    p.add_argument("--cache", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_path}")

    print(f"Building model: {args.model} (fine-tuning from COCO-pretrained weights)")
    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        resume=True,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        cache=args.cache,

        # Augmentation tuned for pre-aligned PCB photographs
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.0,

        # Optimizer settings
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        close_mosaic=10,
        cls=args.cls_weight,

        # Saving Settings
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    print("\nTraining complete.")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")

if __name__ == "__main__":
    main()
