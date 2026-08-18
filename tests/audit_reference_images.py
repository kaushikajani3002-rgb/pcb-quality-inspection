import os
import sys
import json
import math
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.detection_engine import normalize_component_coordinates, map_class_to_component_type

def audit_image(image_path, template_path, model):
    if not os.path.exists(image_path) or not os.path.exists(template_path):
        print(f"Skipping {image_path} / {template_path} (file missing)")
        return
        
    print("=" * 80)
    print(f"AUDITING: {os.path.basename(image_path)} vs {os.path.basename(template_path)}")
    print("=" * 80)
    
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    with open(template_path, "r") as f:
        template = json.load(f)
        
    board_dims = template.get("board_dimensions", {})
    t_w_mm = board_dims.get("width_mm", 100.0)
    t_h_mm = board_dims.get("height_mm", 100.0)
    
    raw_expected = template.get("components", [])
    expected_comps = [normalize_component_coordinates(c, w, h) for c in raw_expected]
    
    print(f"Template Name: {template.get('board_name')} | Image Res: {w}x{h} | Expected Count: {len(expected_comps)}")
    
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    boxes = results[0].boxes
    
    print(f"YOLO Raw Detections Count (conf=0.25): {len(boxes)}")
    
    detected_comps = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        c_val = float(box.conf[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        if "pcb" in class_name.lower():
            continue
            
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
        
    print("\n--- DETECTED COMPONENTS ---")
    for dc in detected_comps:
        print(f"  [{dc['id']}] Class='{dc['class_name']}' Type='{dc['type']}' Conf={dc['confidence']:.2f} CenterPct=({dc['center_x_pct']:.2f}%, {dc['center_y_pct']:.2f}%)")
        
    print("\n--- EXPECTED MATCH ANALYSIS ---")
    matched_exp_ids = set()
    unmatched_dets = []
    
    for dc in detected_comps:
        dc_x_mm = (dc["center_x_pct"] / 100.0) * t_w_mm
        dc_y_mm = (dc["center_y_pct"] / 100.0) * t_h_mm
        
        best_exp = None
        best_dist = 999.0
        
        for ec in expected_comps:
            ec_x_mm = (ec["center_x_pct"] / 100.0) * t_w_mm
            ec_y_mm = (ec["center_y_pct"] / 100.0) * t_h_mm
            dist = math.sqrt((ec_x_mm - dc_x_mm) ** 2 + (ec_y_mm - dc_y_mm) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_exp = ec
                
        if best_exp and best_dist <= 15.0:
            matched_exp_ids.add(best_exp["id"])
            print(f"  ✔ {dc['id']} ('{dc['class_name']}') MATCHED -> '{best_exp['id']}' ({best_exp['type']}) | Dist: {best_dist:.2f} mm")
        else:
            unmatched_dets.append((dc, best_exp, best_dist))
            nearest_info = f"Nearest: '{best_exp['id']}' at {best_dist:.2f} mm" if best_exp else "No expected near"
            print(f"  ⚠️ {dc['id']} ('{dc['class_name']}') UNMATCHED / EXTRA | {nearest_info}")

    missing_exp_ids = [ec["id"] for ec in expected_comps if ec["id"] not in matched_exp_ids]
    print(f"\nSummary for {os.path.basename(image_path)}:")
    print(f"  - Total Expected: {len(expected_comps)}")
    print(f"  - Matched Expected: {len(matched_exp_ids)}")
    print(f"  - Missing Expected: {len(missing_exp_ids)} -> {missing_exp_ids}")
    print(f"  - Total Detected: {len(detected_comps)}")
    print(f"  - Unmatched Detections (Extra Candidates): {len(unmatched_dets)}")
    print("\n")

def main():
    model_path = r"D:\PCB\PCB_AOI\models\trained\Component\Component_best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
        
    model = YOLO(model_path)
    
    audit_image(r"D:\PCB\New folder\Arduino-uno.jpg", r"D:\PCB\PCB_AOI\templates\arduino_uno.json", model)
    audit_image(r"D:\PCB\New folder\ESP32-DEV-KIT-V1-pin.jpg", r"D:\PCB\PCB_AOI\templates\esp32_devkit.json", model)
    audit_image(r"D:\PCB\New folder\STM32-Blue-Pill.jpg", r"D:\PCB\PCB_AOI\templates\stm32_blue_pill.json", model)

if __name__ == "__main__":
    main()
