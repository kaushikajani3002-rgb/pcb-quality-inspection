import os
import sys
import json
import math
from pathlib import Path
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.model_manager import ModelManager
from src.ai.detection_engine import run_component_inspection, normalize_component_coordinates

def main():
    image_path = r"D:\PCB\New folder\Arduino-uno.jpg"
    template_path = r"D:\PCB\PCB_AOI\templates\arduino_uno.json"
    
    if not os.path.exists(image_path) or not os.path.exists(template_path):
        print("Missing required image or template file")
        return
        
    img = Image.open(image_path).convert("RGB")
    w_orig, h_orig = img.size
    
    with open(template_path, "r") as f:
        template = json.load(f)
        
    mm = ModelManager()
    model = mm.get_component_model()
    
    # Run full component inspection pipeline
    comp_res = run_component_inspection(
        uploaded_image=img,
        component_model=model,
        conf_slider=0.25,
        iou_slider=0.45,
        active_template=template,
        position_tolerance_slider=1.5
    )
    
    debug_info = comp_res.get("debug_info", {})
    raw_dets = debug_info.get("raw_detections", [])
    expected_comps = template.get("components", [])
    board_dims = template.get("board_dimensions", {})
    w_mm = float(board_dims.get("width_mm", 68.6))
    h_mm = float(board_dims.get("height_mm", 53.4))
    
    print("================================================================================")
    print("1. ALL RAW YOLO DETECTIONS ON ORIGINAL IMAGE SPACE")
    print("================================================================================")
    print(f"Total raw detections: {len(raw_dets)}")
    for i, rd in enumerate(raw_dets):
        print(f"  [{i+1:02d}] Class='{rd.get('class_name')}' | Type='{rd.get('type')}' | Conf={rd.get('confidence', 0):.4f} | BBox=[{rd.get('x1'):.1f}, {rd.get('y1'):.1f}, {rd.get('x2'):.1f}, {rd.get('y2'):.1f}] | CenterPct=({rd.get('center_x_pct'):.2f}%, {rd.get('center_y_pct'):.2f}%)")

    print("\n================================================================================")
    print("2. ALL EXPECTED TEMPLATE COMPONENTS")
    print("================================================================================")
    print(f"Total expected components: {len(expected_comps)}")
    for i, ec in enumerate(expected_comps):
        cx_pct = float(ec["center_x_pct"]) * 100.0 if float(ec["center_x_pct"]) <= 1.0 else float(ec["center_x_pct"])
        cy_pct = float(ec["center_y_pct"]) * 100.0 if float(ec["center_y_pct"]) <= 1.0 else float(ec["center_y_pct"])
        print(f"  [{i+1:02d}] ID='{ec.get('id')}' | Type='{ec.get('type')}' | ExpectedCenterPct=({cx_pct:.2f}%, {cy_pct:.2f}%)")

    print("\n================================================================================")
    print("3. DETAILED EXPECTED COMPONENT MATCH DIAGNOSTIC TABLE")
    print("================================================================================")
    header = f"{'Expected ID':25s} | {'Expected Type':12s} | {'Detected Class':18s} | {'Dist (mm)':10s} | {'Match Status':14s} | Reason"
    print(header)
    print("-" * 115)
    
    matched_dets = comp_res.get("detected_components", [])
    missing_list = comp_res.get("missing", [])
    extra_list = comp_res.get("extra", [])
    
    matched_map = {d["id"]: d for d in matched_dets if d.get("status") in ["Correct", "Misaligned"]}
    
    matched_exp_count = 0
    unmatched_exp_count = 0
    
    for ec in expected_comps:
        eid = ec["id"]
        etype = ec["type"]
        ec_x_mm = (float(ec["center_x_pct"]) if float(ec["center_x_pct"]) <= 1.0 else float(ec["center_x_pct"])/100.0) * w_mm
        ec_y_mm = (float(ec["center_y_pct"]) if float(ec["center_y_pct"]) <= 1.0 else float(ec["center_y_pct"])/100.0) * h_mm
        
        closest_det = None
        closest_dist = 999.0
        for rd in raw_dets:
            rd_x_mm = (rd["center_x_pct"] / 100.0) * w_mm
            rd_y_mm = (rd["center_y_pct"] / 100.0) * h_mm
            dist = math.sqrt((ec_x_mm - rd_x_mm) ** 2 + (ec_y_mm - rd_y_mm) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_det = rd
                
        if eid in matched_map:
            m_det = matched_map[eid]
            matched_exp_count += 1
            status_str = "MATCHED"
            reason_str = f"Paired with detection '{m_det.get('class_name')}'"
            det_class = m_det.get("class_name", "N/A")
            dist_val = f"{m_det.get('distance_mm', 0):.2f} mm"
        else:
            unmatched_exp_count += 1
            status_str = "UNMATCHED"
            det_class = closest_det.get("class_name") if closest_det else "None"
            dist_val = f"{closest_dist:.2f} mm" if closest_det else "N/A"
            
            if not closest_det or closest_dist > 25.0:
                reason_str = "YOLO did not detect component (No bbox near expected location)"
            elif str(closest_det.get("type")).upper() != str(etype).upper():
                reason_str = f"Type mismatch: expected '{etype}', closest detected was '{closest_det.get('class_name')}' ('{closest_det.get('type')}')"
            else:
                reason_str = f"Spatial distance ({closest_dist:.2f}mm) exceeds tolerance or assigned to another expected component"
                
        print(f"{eid:25s} | {etype:12s} | {det_class:18s} | {dist_val:10s} | {status_str:14s} | {reason_str}")

    print("\n================================================================================")
    print("4. SUMMARY METRICS")
    print("================================================================================")
    print(f"Total YOLO detections        : {len(raw_dets)}")
    print(f"Matched detections          : {len(matched_map)}")
    print(f"Unmatched detections (Extra): {len(extra_list)}")
    print(f"Expected components         : {len(expected_comps)}")
    print(f"Matched expected            : {matched_exp_count}")
    print(f"Unmatched expected (Missing): {unmatched_exp_count}")

if __name__ == "__main__":
    main()
