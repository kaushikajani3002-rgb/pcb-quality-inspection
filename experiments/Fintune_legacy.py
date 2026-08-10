import os
import torch
from ultralytics import YOLO

def main():
    print("🚀 Initializing Fine-Tuning Pipeline...")

    # 1. Define Paths
    pretrained_weights_path = r"D:\Pcb Quality Inspection\outputs\pcb_components-2\weights\best.pt"
    yaml_data_path = r"D:\Pcb Quality Inspection\infarance\ALL_cercit\data.yaml"
    output_dir = r"D:\Pcb Quality Inspection\infarance\ALL_cercit\output"

    # Verify files exist
    if not os.path.exists(pretrained_weights_path):
        print(f"❌ Error: Pre-trained weights file not found at: {pretrained_weights_path}")
        return

    if not os.path.exists(yaml_data_path):
        print(f"❌ Error: Dataset YAML configuration not found at: {yaml_data_path}")
        return

    # 2. Load Model Weights
    print(f"📦 Loading pre-trained model weights from: {pretrained_weights_path}")
    model = YOLO(pretrained_weights_path)

    # 3. Optimized Fine-Tuning Process
    print("⚡ Starting Fine-Tuning with optimized hyperparameters...")
    
    results = model.train(
        data=yaml_data_path,
        epochs=120,                  # Increased: Merged multi-source datasets need ~120+ epochs
        imgsz=640,
        batch=8,                    # Increase to 16 if VRAM allows
        
        # Hardware Allocation
        device=-1,                   # Auto-select idle GPU
        workers=4,
        
        # Fine-Tuning & Learning Rate Schedule
        lr0=0.003,                  # Optimized initial LR for cross-domain learning
        lrf=0.01,                   # Final LR fraction
        cos_lr=True,                # CRITICAL: Enables smooth cosine decay
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=25,                # Prevents early stopping before convergence
        
        # FIXED: Removed resume=True so custom parameters actually apply
        resume=False,
        cache=False,

        # Loss Function Weights
        cls=1.0,                    # Restores classification confidence
        
        # PCB-Specific Optimized Augmentations
        degrees=15.0,               # Reduced from 180.0 to protect small defect details
        scale=0.2,                  # Moderate scale jitter
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15,            # CRITICAL: Disables mosaic for final 15 epochs to stabilize boxes
        
        # Logging & Model Saving
        project=output_dir,
        name="PCB_FineTuned_Model_V2",
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n✅ Fine-tuning completed successfully!")
    fine_tuned_weights = os.path.join(output_dir, "PCB_FineTuned_Model_V2", "weights", "best.pt")
    print(f"🎯 Fine-tuned weights saved at: {fine_tuned_weights}")

if __name__ == "__main__":
    main()