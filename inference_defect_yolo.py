"""
YOLO11m Batch Inference Script -- PCB Defect Detection
================================================================
Runs batch inference on a folder of target PCB images using fine-tuned YOLO11m weights 
and generates output files along with classification scores.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Class mapping (0 to 8) for typical PCB defects
PCB_DEFECT_CLASSES = {
    0: "SH (Short)",
    1: "SP (Spur)",
    2: "SC (Spurious Copper)",
    3: "OP (Open)",
    4: "MB (Mouse Bite)",
    5: "HB (Hole Breakout)",
    6: "CS (Conductor Scratch)",
    7: "CFO (Conductor Foreign Object)",
    8: "BMFO (Base Material Foreign Object)"
}

def main():
    # 1. Define Model and Input Paths
    model_path = r"models/dspcbsd.pt"  # Update with your best.pt path
    input_folder = r"datasets/inferance/Images"       # Folder containing test PCB images
    output_folder = r"outputs/inferance_results"     # Folder where results will be saved

    # Verify model weights exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights file not found at: {model_path}")
        return

    # Verify input images folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input images folder not found at: {input_folder}")
        return

    # Initialize YOLO Model
    print(f"📦 Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # 2. Run Batch Inference on Folder
    print(f"⚡ Running batch inference on: {input_folder}")
    results = model.predict(
        source=input_folder,
        conf=0.25,             # Confidence threshold
        iou=0.45,              # NMS IoU threshold
        imgsz=640,             # Image resolution (matches training size)
        project=output_folder, # Parent directory for outputs
        name="results",        # Subfolder inside project
        exist_ok=True,         # Overwrite or append to existing directory
        save=True,             # Save annotated images
        save_txt=True,         # Save detection bounding box coordinates to .txt files
        save_conf=True,        # Include confidence scores in .txt files
        device="cpu"           # Set to "0" for GPU or "cpu"
    )

    # 3. Print Detections Summary Report
    print(f"\n==========================================")
    print(f" Batch Inference Complete!")
    print(f" Saved Results to: {os.path.join(output_folder, 'results')}")
    print(f"==========================================")

    total_defects = 0
    for result in results:
        img_name = Path(result.path).name
        boxes = result.boxes
        num_defects = len(boxes)
        total_defects += num_defects
        
        if num_defects > 0:
            print(f"\n📄 Image: {img_name} -> Found {num_defects} defect(s):")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_label = PCB_DEFECT_CLASSES.get(cls_id, f"Class {cls_id}")
                print(f"   • {class_label:<30} | Confidence: {conf:.2f}")
        else:
            print(f"📄 Image: {img_name} -> No defects detected.")

    print(f"\nTotal defects detected across all images: {total_defects}")

if __name__ == "__main__":
    main()
