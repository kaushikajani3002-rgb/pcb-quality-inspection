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

def project_coordinates(comp: Dict, W: float, H: float, board_box: list) -> dict:
    c = comp.copy()
    c.setdefault("id", "Unknown")
    c.setdefault("type", "Unknown")
    c.setdefault("class_id", -1)
    c.setdefault("class_name", c["type"])
    c.setdefault("confidence", 1.0)
    
    cx_pct = float(c.get("center_x_pct", 0.5))
    cy_pct = float(c.get("center_y_pct", 0.5))
    w_pct = float(c.get("width_pct", 0.1))
    h_pct = float(c.get("height_pct", 0.1))
    
    # Check if ratio or percentage
    is_ratio = False
    for val in [cx_pct, cy_pct, w_pct, h_pct]:
        if 0 < val <= 1.0:
            is_ratio = True
            break
            
    if is_ratio:
        cx_ratio = cx_pct
        cy_ratio = cy_pct
        w_ratio = w_pct
        h_ratio = h_pct
    else:
        cx_ratio = cx_pct / 100.0
        cy_ratio = cy_pct / 100.0
        w_ratio = w_pct / 100.0
        h_ratio = h_pct / 100.0
        
    bx1, by1, bx2, by2 = board_box
    bw = bx2 - bx1
    bh = by2 - by1
    
    cx = bx1 + cx_ratio * bw
    cy = by1 + cy_ratio * bh
    width = w_ratio * bw
    height = h_ratio * bh
    
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0
    
    c["x1"] = round(x1, 2)
    c["y1"] = round(y1, 2)
    c["x2"] = round(x2, 2)
    c["y2"] = round(y2, 2)
    c["center_x"] = round(cx, 2)
    c["center_y"] = round(cy, 2)
    c["width"] = round(width, 2)
    c["height"] = round(height, 2)
    c["center_x_pct"] = round((cx / W) * 100.0, 2)
    c["center_y_pct"] = round((cy / H) * 100.0, 2)
    c["width_pct"] = round((width / W) * 100.0, 2)
    c["height_pct"] = round((height / H) * 100.0, 2)
    
    return c

def main():
    image_path = r"D:\PCB\New folder\Arduino-uno.jpg"
    template_path = r"D:\PCB\PCB_AOI\templates\arduino_uno.json"
    model_path = r"D:\PCB\PCB_AOI\models\trained\Component\All_cercit_finetuned_best.pt"
    
    # 1. Load template
    with open(template_path, "r") as f:
        template = json.load(f)
    board_dims = template.get("board_dimensions", {})
    t_w_mm = board_dims.get("width_mm", 68.6)
    t_h_mm = board_dims.get("height_mm", 53.4)
    expected_raw = template.get("components", [])
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Load model
    model = YOLO(model_path)
    
    # Run YOLO to find PCB
    print("--- Running YOLO to locate PCB Board ---")
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    boxes = results[0].boxes
    pcb_box = None
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        if class_name == "PCB":
            pcb_box = box.xyxy[0].tolist()
            print(f"Found PCB board bounding box: {pcb_box}")
            break
            
    if not pcb_box:
        print("PCB not found!")
        return
        
    # Project expected coordinates onto PCB board box
    expected_comps = [project_coordinates(c, w, h, pcb_box) for c in expected_raw]
    
    # Normalize detections (run conf=0.15)
    detected_comps = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        c_val = float(box.conf[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy
        mapped_type = map_class_to_component_type(class_name)
        
        norm_det = normalize_component_coordinates({
            "id": f"DET_{i+1}",
            "class_id": cls_id,
            "class_name": class_name,
            "type": mapped_type,
            "confidence": c_val,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }, w, h)
        detected_comps.append(norm_det)

    print("\n--- Projective Mapping Pairwise Distance Matrix ---")
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
