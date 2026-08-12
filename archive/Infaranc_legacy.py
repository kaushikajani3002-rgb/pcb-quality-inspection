import os
from pathlib import Path
import cv2
from ultralytics import YOLO

# 1. Load your trained YOLOv11m weights
model_path = r"D:\Pcb Quality Inspection\infarance\ALL_cercit\output\PCB_FineTuned_Model_V2\weights\best.pt"  # Replace with your best.pt path
model = YOLO(model_path)

# 2. Define input image folder and custom output folder
input_folder = r"D:\Pcb Quality Inspection\infarance\ALL_cercit\test\images"       # Folder containing test PCB images
output_folder = r"D:\Pcb Quality Inspection\infarance\ALL_cercit\output\Infarance"    # Folder where results will be saved

# 3. Exact DsPCBSD class mapping (0 to 8)
DSPCBSD_CLASSES = {
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

# 4. Run batch inference on the entire folder
results = model.predict(
    source=input_folder,
    conf=0.7,             # Confidence threshold
    iou=0.45,              # NMS IoU threshold
    imgsz=640,             # Image resolution (match training size)
    project=output_folder, # Parent directory for outputs
    name="results",        # Subfolder inside project
    exist_ok=True,         # Overwrite or append to existing directory without creating results2, results3...
    save=True,             # Save annotated images
    save_txt=True,         # Save detection bounding box coordinates to .txt files
    save_conf=True,        # Include confidence scores in .txt files
    device=0               # Use GPU '0' or 'cpu'
)

# 5. Summary Log of Detections Across All Images
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
            class_label = DSPCBSD_CLASSES.get(cls_id, f"Class {cls_id}")
            print(f"   • {class_label:<30} | Confidence: {conf:.2f}")
    else:
        print(f"📄 Image: {img_name} -> No defects detected.")

print(f"\nTotal defects detected across all images: {total_defects}")