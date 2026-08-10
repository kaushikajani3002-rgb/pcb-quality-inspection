"""
YOLO11m Fine-Tuning Pipeline -- PCB Defect Detection
================================================================
Fine-tunes pre-trained YOLO11m weights on a merged trace defect dataset
(such as DeepPCB or DsPCBSD+) to classify and segment board-level faults.
"""

import os
from ultralytics import YOLO

def main():
    print("🚀 Initializing Fine-Tuning Pipeline...")

    # 1. Define Paths (Update these paths to match your local setup)
    pretrained_weights_path = r"models/component_yolo11m.pt"
    yaml_data_path = r"datasets/merged/data.yaml"
    output_dir = r"runs/detect"

    # Verify that the base weights exist before starting
    if not os.path.exists(pretrained_weights_path):
        print(f"⚠️ Warning: Pre-trained weights file not found at: {pretrained_weights_path}. Will download default weights.")
        pretrained_weights_path = "yolo11m.pt"

    # Verify that data.yaml exists
    if not os.path.exists(yaml_data_path):
        print(f"❌ Error: Dataset YAML configuration not found at: {yaml_data_path}")
        return

    # 2. Load Model from Existing Weights
    print(f"📦 Loading pre-trained model weights from: {pretrained_weights_path}")
    model = YOLO(pretrained_weights_path)

    # 3. Start Fine-Tuning Process
    print("⚡ Starting Fine-Tuning with idle GPU auto-selection (device=-1)...")
    
    results = model.train(
        data=yaml_data_path,
        epochs=70,                  # Fine-tuning epoch count
        imgsz=640,                  # Image resolution
        batch=8,                    # Adjust based on your GPU VRAM (e.g. 8 or 4 if facing OOM)
        device=-1,                  # Auto-selects the most idle GPU
        workers=4,                  # CPU threads allocated for pre-processing
        
        # Fine-Tuning Hyperparameters
        lr0=0.001,                   # Lower initial learning rate for fine-tuning stability
        lrf=0.01,                    # Final learning rate ratio
        momentum=0.937,              # Optimizer momentum
        weight_decay=0.0005,         # L2 Weight Decay
        warmup_epochs=2.0,           # Warmup epochs for gradient stabilization
        resume=True,
        patience=20,
        cache=False,
        
        # PCB-Specific Augmentations
        degrees=180.0,               # Full 360° rotational invariance for PCBs
        flipud=0.5,                  # Vertical flip ratio
        fliplr=0.5,                  # Horizontal flip ratio
        mosaic=1.0,                  # Mosaic augmentation for small features/cracks
        
        # Logging & Model Saving
        project=output_dir,
        name="PCB_FineTuned_Model",   # Output folder name
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n✅ Fine-tuning completed successfully!")
    fine_tuned_weights = os.path.join(output_dir, "PCB_FineTuned_Model", "weights", "best.pt")
    print(f"🎯 Fine-tuned weights saved at: {fine_tuned_weights}")

if __name__ == "__main__":
    main()
