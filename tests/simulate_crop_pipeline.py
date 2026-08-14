import os
import sys
import json
import math
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Insert project root into sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.detection_engine import normalize_component_coordinates, map_class_to_component_type

def main():
    image_path = r"D:\PCB\New folder\Arduino-uno.jpg"
    template_path = r"D:\PCB\PCB_AOI\templates\arduino_uno.json"
    model_path = r"D:\PCB\PCB_AOI\models\trained\Component\All_cercit_finetuned_best.pt"
    
    # 1. Load template
    with open(template_path, "r") as f:
        template = json.load(f)
    board_dims = template.get("board_dimensions", {})
    t_w = board_dims.get("pixel_width", 640)
    t_h = board_dims.get("pixel_height", 480)
    t_w_mm = board_dims.get("width_mm", 68.6)
    t_h_mm = board_dims.get("height_mm", 53.4)
    expected_raw = template.get("components", [])
    
    # Load original image
    img = Image.open(image_path).convert("RGB")
    w_orig, h_orig = img.size
    
    # Load model
    model = YOLO(model_path)
    
    # 2. Run first pass to locate PCB board boundary
    print("--- STEP 1: Running YOLO to locate PCB Board ---")
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    boxes = results[0].boxes
    pcb_box = None
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        if class_name == "PCB":
            pcb_box = box.xyxy[0].tolist()
            print(f"Found PCB board bounding box: {pcb_box} (conf={float(box.conf[0]):.4f})")
            break
            
    if not pcb_box:
        print("PCB board not detected in target image! Cannot proceed with alignment.")
        return
        
    # 3. Crop PCB board and resize to template resolution (640x480)
    print("\n--- STEP 2: Cropping and Resizing PCB board to template resolution (640x480) ---")
    x1, y1, x2, y2 = pcb_box
    cropped_img = img.crop((x1, y1, x2, y2))
    normalized_img = cropped_img.resize((t_w, t_h), Image.Resampling.LANCZOS)
    
    # Save cropped image for manual inspection if needed
    normalized_img.save("normalized_pcb_board.jpg")
    print("Saved cropped normalized PCB board image as normalized_pcb_board.jpg")
    
    # 4. Run component detection on the normalized PCB board
    print("\n--- STEP 3: Running component detection on normalized PCB image ---")
    results_norm = model.predict(source=normalized_img, conf=0.15, imgsz=640, verbose=False)
    boxes_norm = results_norm[0].boxes
    print(f"Total components detected on normalized PCB at conf=0.15: {len(boxes_norm)}")
    
    detected_comps = []
    for i, box in enumerate(boxes_norm):
        cls_id = int(box.cls[0])
        c_val = float(box.conf[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        xyxy = box.xyxy[0].tolist()
        dx1, dy1, dx2, dy2 = xyxy
        
        # Apply class mapping
        mapped_type = map_class_to_component_type(class_name)
        
        # Since the image is now normalized to template size (t_w x t_h), 
        # the coordinates are normalized against (t_w, t_h).
        norm_det = normalize_component_coordinates({
            "id": f"DET_{i+1}",
            "class_id": cls_id,
            "class_name": class_name,
            "type": mapped_type,
            "confidence": c_val,
            "x1": dx1,
            "y1": dy1,
            "x2": dx2,
            "y2": dy2
        }, t_w, t_h)
        detected_comps.append(norm_det)
        
        print(f"  [{i+1}] Class='{class_name}', Type='{mapped_type}', Conf={c_val:.4f}, BBox=[{dx1:.1f}, {dy1:.1f}, {dx2:.1f}, {dy2:.1f}], Pct=({norm_det['center_x_pct']:.2f}, {norm_det['center_y_pct']:.2f})")

    # 5. Spatial template matching trace
    print("\n--- STEP 4: Matching detected components with expected template ---")
    expected_comps = [normalize_component_coordinates(c, t_w, t_h) for c in expected_raw]
    
    print(f"{'Expected ID':25s} | {'Matched ID':12s} | {'Detected Class':20s} | {'Distance (mm)':15s} | {'Result':10s}")
    print("-" * 90)
    
    assigned_dets = set()
    for ec in expected_comps:
        best_det = None
        best_dist = 999.0
        best_idx = None
        
        ec_x_mm = (ec["center_x_pct"] / 100.0) * t_w_mm
        ec_y_mm = (ec["center_y_pct"] / 100.0) * t_h_mm
        
        for idx, dc in enumerate(detected_comps):
            if idx in assigned_dets:
                continue
            if ec["type"] != dc["type"]:
                continue
                
            dc_x_mm = (dc["center_x_pct"] / 100.0) * t_w_mm
            dc_y_mm = (dc["center_y_pct"] / 100.0) * t_h_mm
            dist = math.sqrt((ec_x_mm - dc_x_mm) ** 2 + (ec_y_mm - dc_y_mm) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_det = dc
                best_idx = idx
                
        # Proximity threshold of 15 mm
        if best_det and best_dist <= 15.0:
            assigned_dets.add(best_idx)
            print(f"{ec['id']:25s} | {best_det['id']:12s} | {best_det['class_name']:20s} | {best_dist:10.2f} mm    | MATCH")
        else:
            # Let's print the closest candidate even if it's outside threshold to see misalignment
            closest_det = None
            closest_dist = 999.0
            for idx, dc in enumerate(detected_comps):
                if ec["type"] == dc["type"]:
                    dc_x_mm = (dc["center_x_pct"] / 100.0) * t_w_mm
                    dc_y_mm = (dc["center_y_pct"] / 100.0) * t_h_mm
                    dist = math.sqrt((ec_x_mm - dc_x_mm) ** 2 + (ec_y_mm - dc_y_mm) ** 2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_det = dc
            if closest_det:
                print(f"{ec['id']:25s} | {closest_det['id']:12s} | {closest_det['class_name']:20s} | {closest_dist:10.2f} mm    | MISALIGNED (dist > 15mm)")
            else:
                print(f"{ec['id']:25s} | {'N/A':12s} | {'N/A':20s} | {'N/A':15s} | MISSING")

if __name__ == "__main__":
    main()
