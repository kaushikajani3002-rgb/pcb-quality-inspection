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
    model_path = r"D:\PCB\PCB_AOI\models\trained\Component\Component_best.pt"
    
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return
    if not os.path.exists(template_path):
        print(f"Template not found at: {template_path}")
        return
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return

    # 1. Check Image Geometry
    img = Image.open(image_path)
    w_orig, h_orig = img.size
    aspect_ratio = w_orig / h_orig
    print("=== 1. IMAGE GEOMETRY ===")
    print(f"Image Path: {image_path}")
    print(f"Original Resolution: {w_orig}x{h_orig}")
    print(f"Aspect Ratio: {aspect_ratio:.4f}")
    print(f"Image Mode: {img.mode}")
    
    # 2. Load Template Info
    with open(template_path, "r") as f:
        template = json.load(f)
    print("\n=== 2. TEMPLATE INFORMATION ===")
    print(f"Board Name: {template.get('board_name')}")
    board_dims = template.get("board_dimensions", {})
    t_w = board_dims.get("pixel_width", 640)
    t_h = board_dims.get("pixel_height", 480)
    t_w_mm = board_dims.get("width_mm", 68.6)
    t_h_mm = board_dims.get("height_mm", 53.4)
    print(f"Template Dimensions: {t_w}x{t_h} pixels ({t_w_mm}x{t_h_mm} mm)")
    print(f"Template Aspect Ratio: {t_w / t_h:.4f}")
    
    expected_raw = template.get("components", [])
    expected_comps = [normalize_component_coordinates(c, w_orig, h_orig) for c in expected_raw]
    print(f"Number of expected components in template: {len(expected_comps)}")

    # 3. Run YOLO Component Inference at different confidence thresholds
    model = YOLO(model_path)
    print("\n=== 3. YOLO INFERENCE DETAILS ===")
    for conf in [0.25, 0.50, 0.70]:
        results = model.predict(source=img, conf=conf, imgsz=640, verbose=False)
        boxes = results[0].boxes
        print(f"Detections at conf={conf:.2f}: {len(boxes)}")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            c_val = float(box.conf[0])
            class_name = model.names.get(cls_id, f"Class {cls_id}")
            xyxy = box.xyxy[0].tolist()
            print(f"  [{i+1}] '{class_name}' conf={c_val:.4f} bbox={xyxy}")

    # Let's run a detailed match trace at conf=0.25 to see why things fail or succeed
    print("\n=== 4. DETAILED MATCHING TRACE AT conf=0.25 ===")
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    boxes = results[0].boxes
    
    # Normalize detections
    detected_comps = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        c_val = float(box.conf[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy
        
        # Apply class mapping
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
        }, w_orig, h_orig)
        detected_comps.append(norm_det)
        
    print(f"Total normalized detections: {len(detected_comps)}")
    
    # Print expected and detected components
    print("\nExpected Components:")
    for ec in expected_comps:
        print(f"  ID={ec['id']:25s} Type={ec['type']:10s} CenterPct=({ec['center_x_pct']:.2f}, {ec['center_y_pct']:.2f}) Pixels=({ec['center_x']:.1f}, {ec['center_y']:.1f})")
        
    print("\nDetected Components:")
    for dc in detected_comps:
        print(f"  Class={dc['class_name']:20s} Type={dc['type']:10s} CenterPct=({dc['center_x_pct']:.2f}, {dc['center_y_pct']:.2f}) Pixels=({dc['center_x']:.1f}, {dc['center_y']:.1f})")

    # Let's compute distances between ALL expected components and ALL detected components of the same mapped type
    print("\n=== 5. PAIRWISE SPATIAL DISTANCE MATRIX ===")
    print(f"{'Expected ID':25s} | {'Detected Class':20s} | {'Distance (mm)':15s} | {'Overlap (IoU)':15s} | {'Compatible?':12s}")
    print("-" * 95)
    
    for ec in expected_comps:
        for dc in detected_comps:
            # Check type compatibility
            type_compat = (ec["type"] == dc["type"])
            
            # Distance in mm
            ec_x_mm = (ec["center_x_pct"] / 100.0) * t_w_mm
            ec_y_mm = (ec["center_y_pct"] / 100.0) * t_h_mm
            dc_x_mm = (dc["center_x_pct"] / 100.0) * t_w_mm
            dc_y_mm = (dc["center_y_pct"] / 100.0) * t_h_mm
            dist_mm = math.sqrt((ec_x_mm - dc_x_mm) ** 2 + (ec_y_mm - dc_y_mm) ** 2)
            
            # IoU
            # Convert both to pixel coords of original image to compute IoU
            ex1, ey1, ex2, ey2 = ec["x1"], ec["y1"], ec["x2"], ec["y2"]
            dx1, dy1, dx2, dy2 = dc["x1"], dc["y1"], dc["x2"], dc["y2"]
            
            inter_x1 = max(ex1, dx1)
            inter_y1 = max(ey1, dy1)
            inter_x2 = min(ex2, dx2)
            inter_y2 = min(ey2, dy2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            else:
                inter_area = 0.0
                
            union_area = (ex2 - ex1) * (ey2 - ey1) + (dx2 - dx1) * (dy2 - dy1) - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            if type_compat or dist_mm < 30.0:
                print(f"{ec['id']:25s} | {dc['class_name']:20s} | {dist_mm:10.2f} mm    | {iou:10.4f}      | {str(type_compat):12s}")

if __name__ == "__main__":
    main()
