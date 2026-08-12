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

This is a DIFFERENT task from trace-defect detection (missing hole / mouse
bite / open / short on a bare board). This model looks at a POPULATED,
assembled board and answers "what component is at each location," which
InspectionEngine then diffs against the expected template layout.

Per README_CLAUDE.md section 6 (Guide for Claude: Integrating YOLO & OpenCV),
this model expects an ALIGNED input image -- i.e. images that have already
gone through inspection/preprocessor.py's homography warp
(cv2.warpPerspective) to match the template's canonical board dimensions.
Train on images captured/warped the same way inference images will be, or
the model will learn a distribution shift it won't see in production.

Two consequences of that alignment step for training config, vs. the earlier
trace-defect script:
  1. Augmentation is toned down. Since inference images are pre-aligned to
     a canonical orientation, heavy geometric augmentation (large rotation,
     vertical flip) would teach the model to expect variation it will never
     actually see, and could hurt precision. Small rotation/translation only
     -- covers residual homography error, not full pose variation.
  2. Color jitter is back ON (hsv_*). Unlike binarized trace images, these
     are real photographs of a populated board under real lighting -- color
     and reflectance vary meaningfully across components (solder, plastic
     packages, silkscreen) and across camera/lighting setups.

Requirements:
    pip install ultralytics --upgrade

This script does NOT run inside this chat's sandbox (no GPU / no internet
here). Run it on your own machine, a Colab GPU runtime, or any cloud GPU box
where your dataset actually lives.

Your data.yaml `names:` list should match the component "type" values used
in your templates/*.json and in detected_components dicts (e.g. IC,
Connector, LED, Crystal, USB_Port, Barrel_Jack, Voltage_Regulator,
Reset_Button, Header_Strip, ...) plus an EXTRA/unregistered class if you
want the model itself to flag unrecognized parts, rather than leaving that
purely to extra_checker.py's downstream diffing logic.

Usage:
    python train_component_detect.py --data /path/to/component_data.yaml
    python train_component_detect.py --data /path/to/component_data.yaml --epochs 150
    python train_component_detect.py --resume runs/detect/pcb_components/weights/last.pt
    python train_component_detect.py --data /path/to/component_data.yaml --output /path/to/my_output_folder
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11m on PCB_AOI component detection dataset")
    p.add_argument("--data", type=str, required=False,
                   default=r"D:\Pcb Quality Inspection\Dataset\archive\All_cercits\data.yaml",
                   help="Path to data.yaml describing train/val/test splits and component classes "
                        "(must match the component 'type' values used in your templates/*.json)")
    p.add_argument("--model", type=str, default="yolo11m.pt",
                   help="Starting weights. COCO-pretrained (yolo11m.pt) fine-tuning is strongly "
                        "recommended here -- an assembled-board dataset from a fixed camera rig is "
                        "almost certainly small (hundreds to low thousands of images), far too small "
                        "for from-scratch training to converge well.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Should match (or be a clean multiple-of-32 crop of) the resolution your "
                        "preprocessor.py warps aligned frames to")
    p.add_argument("--batch", type=int, default=8,
                   help="Lower this (e.g. 8) if you hit CUDA out-of-memory")
    p.add_argument("--device", type=str, default="-1",
                   help="'0' for first GPU, 'cpu' for CPU-only, '0,1' for multi-GPU")
    p.add_argument("--project", type=str, default="All_cercits_output")
    p.add_argument("--name", type=str, default="pcb_components")
    # --- Added: dedicated output folder override ---
    # If provided, this overrides --project as the base folder Ultralytics
    # writes results into (weights/, plots, logs, etc. still land under
    # <output>/<name>/ exactly like the existing project/name behavior).
    # Left as None by default so nothing changes unless you explicitly pass it.
    p.add_argument("--output", type=str,default=r"D:\Pcb Quality Inspection\outputs",
                   help="Folder to store training output in (overrides --project if set). "
                        "Results are saved to <output>/<name>/, e.g. "
                        "--output D:/MyResults --name pcb_components -> D:/MyResults/pcb_components/")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience (epochs with no val improvement)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a last.pt checkpoint to resume an interrupted run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cls-weight", type=float, default=1.0,
                   help="Classification loss weight. Bump this if some component types are rare "
                        "in your dataset (e.g. a board template with only 1 crystal vs. 8 connectors) "
                        "and you see weak recall on those classes after a first run.")
    p.add_argument("--cache", type=str, default=None, choices=[None],
                   help="Cache images in RAM or on disk for faster repeated epochs, if space allows.")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Added: resolve the effective output/project folder ---
    # --output takes priority over --project when explicitly provided;
    # otherwise --project (unchanged) is used exactly as before.
    output_project = args.output if args.output else args.project

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_path}. "
            f"Pass the correct path with --data /path/to/data.yaml"
        )

    print(f"Building model: {args.model} (fine-tuning from COCO-pretrained weights)")
    model = YOLO(args.model)

    print(f"Output folder: {Path(output_project) / args.name}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        resume=True,
        batch=args.batch,
        device=-1,
        project=output_project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        cache=args.cache,

        # --- Augmentation tuned for ALIGNED, populated-board photos ---
        # Color jitter ON: real lighting/reflectance variation across
        # components and capture sessions is meaningful signal to be robust
        # to, unlike the binarized trace-defect case.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Geometric augmentation kept SMALL: inference images are already
        # homography-aligned to the template's canonical orientation, so
        # large rotation/flip would simulate pose variation the model will
        # never actually encounter in production.
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        flipud=0.0,          # boards have a fixed up/down after alignment -- don't flip
        fliplr=0.0,          # silkscreen text + polarized parts (diodes, ICs) have real orientation
        mosaic=1.0,          # still helps with small components
        mixup=0.0,

        # --- Optimizer / schedule ---
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        warmup_epochs=3.0,

        
        close_mosaic=10,
        cls=args.cls_weight,

        # --- Saving ---
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    print("\nTraining complete.")
    print(f"Best weights: {output_project}/{args.name}/weights/best.pt")
    print("Point models/yolo_wrapper.py at this checkpoint (see README_CLAUDE.md section 6, step 2).")

    print("\nRunning validation on test split (if defined in data.yaml)...")
    metrics = model.val(data=str(data_path), split="test")
    print(metrics.box.map)     # mAP50-95
    print(metrics.box.map50)   # mAP50


if __name__ == "__main__":
    main()