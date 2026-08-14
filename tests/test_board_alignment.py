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
    
    # Load template
    with open(template_path, "r") as f:
        template = json.load(f)
    board_dims = template.get("board_dimensions", {})
    t_w = board_dims.get("pixel_width", 640)
    t_h = board_dims.get("pixel_height", 480)
    t_w_mm = board_dims.get("width_mm", 68.6)
    t_h_mm = board_dims.get("height_mm", 53.4)
    expected_raw = template.get("components", [])
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    w_orig, h_orig = img.size
    
    # Load model
    model = YOLO(model_path)
    
    # Step 1: Detect board
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    boxes = results[0].boxes
    pcb_box = None
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        if class_name == "PCB":
            pcb_box = box.xyxy[0].tolist()
            print(f"Detected PCB board boundary: {pcb_box}")
            break
            
    if not pcb_box:
        # Fallback to full image
        pcb_box = [0, 0, w_orig, h_orig]
        print("PCB not detected, using full image boundaries.")
        
    # Step 2: Crop and resize to 640x480
    bx1, by1, bx2, by2 = pcb_box
    bw = bx2 - bx1
    bh = by2 - by1
    cropped_img = img.crop((bx1, by1, bx2, by2))
    normalized_img = cropped_img.resize((t_w, t_h), Image.Resampling.LANCZOS)
    
    # Step 3: Run YOLO on cropped image
    results_norm = model.predict(source=normalized_img, conf=0.15, imgsz=640, verbose=False)
    boxes_norm = results_norm[0].boxes
    
    # Step 4: Map coordinates back to original image space
    detected_comps = []
    for i, box in enumerate(boxes_norm):
        cls_id = int(box.cls[0])
        c_val = float(box.conf[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        if class_name == "PCB":
            continue # skip PCB detection on the cropped image
            
        dx1_norm, dy1_norm, dx2_norm, dy2_norm = box.xyxy[0].tolist()
        
        # Inverse transform to original image pixel coordinates
        dx1_orig = bx1 + (dx1_norm / t_w) * bw
        dy1_orig = by1 + (dy1_norm / t_h) * bh
        dx2_orig = bx1 + (dx2_norm / t_w) * bw
        dy2_orig = by1 + (dy2_norm / t_h) * bh
        
        mapped_type = map_class_to_component_type(class_name)
        
        norm_det = normalize_component_coordinates({
            "id": f"DET_{i+1}",
            "class_id": cls_id,
            "class_name": class_name,
            "type": mapped_type,
            "confidence": c_val,
            "x1": dx1_orig,
            "y1": dy1_orig,
            "x2": dx2_orig,
            "y2": dy2_orig
        }, w_orig, h_orig)
        detected_comps.append(norm_det)
        
        print(f"Detected component {norm_det['id']}: Class='{class_name}', Type='{mapped_type}', CenterPct=({norm_det['center_x_pct']:.2f}, {norm_det['center_y_pct']:.2f}), Pixels=({norm_det['center_x']:.1f}, {norm_det['center_y']:.1f})")

    # Project expected template coordinates onto PCB board box
    expected_comps = []
    for ec in expected_raw:
        cx_pct = float(ec.get("center_x_pct"))
        cy_pct = float(ec.get("center_y_pct"))
        w_pct = float(ec.get("width_pct"))
        h_pct = float(ec.get("height_pct"))
        
        cx_ratio = cx_pct if cx_pct <= 1.0 else cx_pct / 100.0
        cy_ratio = cy_pct if cy_pct <= 1.0 else cy_pct / 100.0
        w_ratio = w_pct if w_pct <= 1.0 else w_pct / 100.0
        h_ratio = h_pct if h_pct <= 1.0 else h_pct / 100.0
        
        cx_orig = bx1 + cx_ratio * bw
        cy_orig = by1 + cy_ratio * bh
        w_orig_comp = w_ratio * bw
        h_orig_comp = h_ratio * bh
        
        x1_orig = cx_orig - w_orig_comp / 2.0
        y1_orig = cy_orig - h_orig_comp / 2.0
        x2_orig = cx_orig + w_orig_comp / 2.0
        y2_orig = cy_orig + h_orig_comp / 2.0
        
        norm_exp = normalize_component_coordinates({
            "id": ec["id"],
            "type": ec["type"],
            "x1": x1_orig,
            "y1": y1_orig,
            "x2": x2_orig,
            "y2": y2_orig
        }, w_orig, h_orig)
        expected_comps.append(norm_exp)
        
    print("\n--- Projective Inverse Mapping Pairwise Distance Matrix ---")
    print(f"{'Expected ID':25s} | {'Matched ID':12s} | {'Detected Class':20s} | {'Distance (mm)':15s} | {'Result':10s}")
    print("-" * 90)
    
    assigned_dets = set()
    for ec in expected_comps:
        best_det = None
        best_dist = 999.0
        best_idx = None
        
        # expected and detected are both in original image coordinates
        # So we can calculate distance in mm using physical millimeters scaled by board dimensions
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
            closest_det = None
            closest_dist = 999.0
            for idx, dc in enumerate(detected_comps):
                if ec["type"] == dc["type"]:
                    dc_x_mm = (dc["center_x_pct"] / 100.0) * t_w_mm
                    dc_y_mm = (dc["center_y_pct"] / 100.0) * t_h_mm
                    dist = math.sqrt((ec_x_mm - dc_x_mm) ** 2 + (dc_y_mm - dc_y_mm) ** 2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_det = dc
            if closest_det:
                print(f"{ec['id']:25s} | {closest_det['id']:12s} | {closest_det['class_name']:20s} | {closest_dist:10.2f} mm    | MISALIGNED (dist > 15mm)")
            else:
                print(f"{ec['id']:25s} | {'N/A':12s} | {'N/A':20s} | {'N/A':15s} | MISSING")

if __name__ == "__main__":
    main()
