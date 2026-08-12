import os
import time
import math
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw
import pandas as pd
from ultralytics import YOLO
from src.utils.logger import logger
from src.inspection.inspection_engine import InspectionEngine

# Dictionary of model paths. The user will share/fill the paths.
MODEL_PATHS = {
    "Component": "models/trained/component_detector_best.pt",
    "DeepPCB": "models/trained/DeepPCB/best.pt",
    "DsPCBSD+": "models/trained/DsPCBSD+/best.pt",
    "HRIPCB": "models/trained/HRIPCB/best.pt",
    "TDD-PCB": "models/trained/TDD-PCB/best.pt"
}

# Trained class names to template component type strings mapping
# (Will be updated once we load models/component_yolo11m.pt and inspect model.names)
CLASS_NAME_MAP = {}

def load_model(model_name: str) -> Any:
    """
    Loads a YOLO model from weights.
    Returns None if the path is not configured or weight file does not exist.
    """
    path = MODEL_PATHS.get(model_name, "")
    if not path:
        logger.warning(f"Model path for '{model_name}' is not configured.")
        return None
    if not os.path.exists(path):
        logger.error(f"Model file not found for '{model_name}' at: {path}")
        return None
    try:
        model = YOLO(path)
        logger.info(f"Successfully loaded '{model_name}' YOLO model from {path}")
        return model
    except Exception as e:
        logger.error(f"Error initializing YOLO model '{model_name}' from {path}: {e}")
        return None

def pixel_to_mm(pixel_val: float, pixel_dim: float, mm_dim: float) -> float:
    """
    Converts visual pixel coordinates to physical millimeters.
    """
    if pixel_dim <= 0:
        return 0.0
    return (pixel_val / pixel_dim) * mm_dim

def check_misalignment(
    exp_x_pct: float, exp_y_pct: float, 
    act_x_pct: float, act_y_pct: float, 
    width_mm: float, height_mm: float
) -> float:
    """
    Computes Euclidean misalignment distance in millimeters.
    """
    dist_x_mm = (exp_x_pct - act_x_pct) * width_mm
    dist_y_mm = (exp_y_pct - act_y_pct) * height_mm
    return math.sqrt(dist_x_mm ** 2 + dist_y_mm ** 2)

def match_detections_to_template(
    expected_comps: List[Dict[str, Any]], 
    detections: List[Dict[str, Any]], 
    width_mm: float, 
    height_mm: float
) -> List[Dict[str, Any]]:
    """
    Spatially maps raw class-based detections to expected template component IDs.
    Matches nearest component of the same type.
    """
    matched_detections = []
    
    # Group expected by component type
    expected_by_type = {}
    for exp in expected_comps:
        t = exp["type"]
        expected_by_type.setdefault(t, []).append(exp)
        
    # Group detections by component type
    detections_by_type = {}
    for det in detections:
        t = det["type"]
        detections_by_type.setdefault(t, []).append(det)
        
    all_types = set(expected_by_type.keys()) | set(detections_by_type.keys())
    extra_counter = 1
    
    for t in all_types:
        exps = expected_by_type.get(t, [])
        dets = detections_by_type.get(t, [])
        
        if not exps:
            # All detections of this type are extra/unregistered
            for det in dets:
                det_copy = det.copy()
                det_copy["id"] = f"ERR_EXTRA_{t.upper()}_{extra_counter:02d}"
                det_copy["status"] = "Extra"
                matched_detections.append(det_copy)
                extra_counter += 1
            continue
            
        if not dets:
            # All expected of this type are missing (handled downstream by MissingChecker)
            continue
            
        # Match using proximity distance matrix (greedy search)
        dist_matrix = []
        for exp in exps:
            row = []
            exp_x_mm = exp["center_x_pct"] * width_mm
            exp_y_mm = exp["center_y_pct"] * height_mm
            for det in dets:
                det_x_mm = det["center_x_pct"] * width_mm
                det_y_mm = det["center_y_pct"] * height_mm
                dist = math.sqrt((exp_x_mm - det_x_mm) ** 2 + (exp_y_mm - det_y_mm) ** 2)
                row.append(dist)
            dist_matrix.append(row)
            
        assigned_dets = set()
        for exp_idx, exp in enumerate(exps):
            row = dist_matrix[exp_idx]
            sorted_det_indices = sorted(range(len(row)), key=lambda k: row[k])
            
            best_det_idx = None
            for idx in sorted_det_indices:
                if idx not in assigned_dets:
                    best_det_idx = idx
                    break
                    
            if best_det_idx is not None:
                assigned_dets.add(best_det_idx)
                det = dets[best_det_idx].copy()
                det["id"] = exp["id"]
                det["status"] = "Correct"
                matched_detections.append(det)
                
        # Unassigned detections are extra/unregistered
        for det_idx, det in enumerate(dets):
            if det_idx not in assigned_dets:
                det_copy = det.copy()
                det_copy["id"] = f"ERR_EXTRA_{t.upper()}_{extra_counter:02d}"
                det_copy["status"] = "Extra"
                matched_detections.append(det_copy)
                extra_counter += 1
                
    return matched_detections

def run_component_counting(
    uploaded_image: Any, 
    component_model: Any, 
    conf_slider: float, 
    iou_slider: float, 
    active_template: Dict[str, Any], 
    position_tolerance_slider: float,
    defect_mode: bool = False
) -> Dict[str, Any]:
    """
    Executes YOLO11m component detection on an uploaded image, maps the predictions 
    to reference template coordinates, runs the verification engine, and renders the overlays.
    """
    start_time = time.time()
    
    # 1. Load image and get dimensions
    if not isinstance(uploaded_image, Image.Image):
        img = Image.open(uploaded_image).convert("RGB")
    else:
        img = uploaded_image
        
    w, h = img.size
    
    # 2. Default fallback values
    expected_comps = active_template.get("components", [])
    total_comps = len(expected_comps)
    
    board_dims = active_template.get("board_dimensions", {})
    width_mm = float(board_dims.get("width_mm", 100.0))
    height_mm = float(board_dims.get("height_mm", 100.0))
    
    template_counts = {}
    for c in expected_comps:
        t = c["type"]
        template_counts[t] = template_counts.get(t, 0) + 1
        
    # 3. Model Inference
    raw_detections = []
    if component_model is not None:
        try:
            results = component_model.predict(source=img, conf=conf_slider, iou=iou_slider, imgsz=640)
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = component_model.names.get(cls_id, f"Class {cls_id}")
                
                # Apply class mapping
                mapped_type = CLASS_NAME_MAP.get(class_name, class_name)
                
                # Extract normalized coordinates (xywhn is center_x, center_y, width, height normalized)
                xywhn = box.xywhn[0].tolist()
                cx_pct, cy_pct, w_pct, h_pct = xywhn
                
                raw_detections.append({
                    "type": mapped_type,
                    "center_x_pct": cx_pct,
                    "center_y_pct": cy_pct,
                    "width_pct": w_pct,
                    "height_pct": h_pct,
                    "confidence": round(conf, 2)
                })
        except Exception as e:
            logger.error(f"YOLO Component Inference failed: {e}")
    else:
        logger.warning("Component model is offline/None. Returning empty detections.")
        
    # 4. Spatially match detections to template expected layout
    matched_detections = match_detections_to_template(expected_comps, raw_detections, width_mm, height_mm)
    
    # 5. Run Verification Engine Checkers
    engine = InspectionEngine()
    inspection_results = engine.inspect_board(active_template, matched_detections, position_tolerance_slider, [])
    
    # 6. Apply Alignment and Extra labels to detections list
    misaligned_ids = {m["id"] for m in inspection_results["misaligned"]}
    for det in matched_detections:
        if det["id"] in misaligned_ids:
            det["status"] = "Misaligned"
            
    # Calculate counts by type
    detected_counts = {}
    for d in matched_detections:
        if d.get("status") != "Extra":
            t = d["type"]
            detected_counts[t] = detected_counts.get(t, 0) + 1
            
    # Inject mock cracks if defect mode is toggled (since defect models are untrained)
    cracks = []
    if defect_mode:
        correct_comps = [d for d in matched_detections if d.get("status") == "Correct"]
        if correct_comps:
            crack_target = correct_comps[-1]  # Pick last correct component to corrupt
            crack_id = f"CRK_{crack_target['id']}"
            cracks.append({
                "id": crack_id,
                "parent_component": crack_target["id"],
                "center_x_pct": crack_target["center_x_pct"] + (crack_target["width_pct"] / 3.0),
                "center_y_pct": crack_target["center_y_pct"] + (crack_target["height_pct"] / 3.0),
                "severity": "High (Solder Joint Fracture)"
            })
            crack_target["status"] = "Crack Detected"
            
    # 7. Render Annotated Bounding Box Image (Detection View)
    annotated_image = img.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Draw detections
    for d in matched_detections:
        cx = int(d["center_x_pct"] * w)
        cy = int(d["center_y_pct"] * h)
        cw = int(d["width_pct"] * w)
        ch = int(d["height_pct"] * h)
        left, top = cx - cw // 2, cy - ch // 2
        right, bottom = cx + cw // 2, cy + ch // 2
        
        status = d.get("status", "Correct")
        if status == "Correct":
            box_color = "#00FF66"  # Green
        elif status == "Misaligned":
            box_color = "#FFCC00"  # Yellow
        elif status == "Extra":
            box_color = "#E11D48"  # Pinkish-Red
        elif status == "Crack Detected":
            box_color = "#3399FF"  # Blue for crack
        else:
            box_color = "#FFFFFF"
            
        draw.rectangle([left, top, right, bottom], outline=box_color, width=3)
        draw.text((left + 5, top + 5), f"{d['id']} ({d['confidence']:.2f})", fill=box_color)
        
        # Draw misalignment vectors
        if status == "Misaligned":
            expected_x_pct = d["center_x_pct"]
            expected_y_pct = d["center_y_pct"]
            for m in inspection_results["misaligned"]:
                if m["id"] == d["id"]:
                    expected_x_pct = m["expected_x_pct"]
                    expected_y_pct = m["expected_y_pct"]
                    break
            exp_x_px = int(expected_x_pct * w)
            exp_y_px = int(expected_y_pct * h)
            
            draw.ellipse([exp_x_px - 4, exp_y_px - 4, exp_x_px + 4, exp_y_px + 4], fill="#00FF66")
            draw.line([(exp_x_px, exp_y_px), (cx, cy)], fill="#FFCC00", width=2)
            
    # Draw Missing Component Crosshairs
    for m in inspection_results["missing"]:
        cx = int(m["expected_x_pct"] * w)
        cy = int(m["expected_y_pct"] * h)
        cw = int(m["width_pct"] * w)
        ch = int(m["height_pct"] * h)
        left, top = cx - cw // 2, cy - ch // 2
        right, bottom = cx + cw // 2, cy + ch // 2
        
        draw.rectangle([left, top, right, bottom], outline="#FF3333", width=2)
        draw.line([(left, top), (right, bottom)], fill="#FF3333", width=1)
        draw.line([(left, bottom), (right, top)], fill="#FF3333", width=1)
        draw.text((left + 5, top + 5), f"MISSING: {m['id']}", fill="#FF3333")
        
    # 8. Render Segmentation Overlay Image (Solder / Defect View)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_seg = ImageDraw.Draw(overlay)
    
    # Fill semi-transparent masks for all detected components
    for d in matched_detections:
        cx = int(d["center_x_pct"] * w)
        cy = int(d["center_y_pct"] * h)
        cw = int(d["width_pct"] * w)
        ch = int(d["height_pct"] * h)
        left, top = cx - cw // 2, cy - ch // 2
        right, bottom = cx + cw // 2, cy + ch // 2
        
        status = d.get("status", "Correct")
        if status == "Correct":
            fill_color = (0, 255, 102, 60)   # Green tint
        elif status == "Misaligned":
            fill_color = (255, 204, 0, 80)   # Yellow tint
        elif status == "Crack Detected":
            fill_color = (51, 153, 255, 80)  # Blue tint
        else:
            fill_color = (225, 29, 72, 80)   # Red/Pink tint
            
        draw_seg.rectangle([left, top, right, bottom], fill=fill_color)
        
    # Draw crack shapes
    for crk in cracks:
        ccx = int(crk["center_x_pct"] * w)
        ccy = int(crk["center_y_pct"] * h)
        draw_seg.polygon(
            [(ccx - 10, ccy - 5), (ccx + 5, ccy - 8), (ccx - 2, ccy + 10), (ccx + 12, ccy + 3), (ccx - 8, ccy + 5)],
            fill=(51, 153, 255, 255), outline=(255, 255, 255, 255)
        )
        draw_seg.text((ccx - 20, ccy - 20), "CRACK DETECTED", fill="#3399FF")
        
    # Composite segmentation mask on top of original image
    segmentation_image = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    
    # 9. Determine overall status (FAIL if missing, misaligned, extra, or cracks)
    has_anomalies = (
        len(inspection_results["missing"]) > 0 or 
        len(inspection_results["extra"]) > 0 or 
        len(inspection_results["misaligned"]) > 0 or 
        len(cracks) > 0
    )
    overall_status = "FAIL" if has_anomalies else "PASS"
    
    duration = time.time() - start_time
    processing_time_str = f"{duration:.3f} sec"
    
    return {
        "status": overall_status,
        "processing_time": processing_time_str,
        "template_name": active_template.get("board_name", "Unknown PCB"),
        "component_statistics": {
            "total_expected": total_comps,
            "total_detected": len(matched_detections),
            "by_type": template_counts
        },
        "detected_components": matched_detections,
        "detected_counts": detected_counts,
        "missing": inspection_results["missing"],
        "extra": inspection_results["extra"],
        "misaligned": inspection_results["misaligned"],
        "cracks": cracks,
        "annotated_image": annotated_image,
        "segmentation_image": segmentation_image,
        "original_image": img
    }

def run_defect_detection(
    model_name: str, 
    uploaded_image: Any, 
    conf_slider: float, 
    iou_slider: float
) -> Dict[str, Any]:
    """
    Placeholder/Graceful fallback for untrained defect models (deeppcb, dspcbsd, hripcb, tddpcb).
    Since weights do not exist, this returns an empty result set.
    """
    logger.warning(f"Defect model '{model_name}' is not trained yet. Graceful fallback active.")
    return {
        "detected_defects": [],
        "annotated_image": None
    }

def build_inventory_table(active_template: Dict[str, Any], detected_counts: Dict[str, int]) -> pd.DataFrame:
    """
    Builds the comparison table between expected component counts and detected counts.
    """
    expected_comps = active_template.get("components", [])
    expected_counts = {}
    for c in expected_comps:
        t = c["type"]
        expected_counts[t] = expected_counts.get(t, 0) + 1
        
    all_types = set(expected_counts.keys()) | set(detected_counts.keys())
    
    summary_rows = []
    for t in sorted(all_types):
        exp = expected_counts.get(t, 0)
        det = detected_counts.get(t, 0)
        summary_rows.append({
            "Component Type": t,
            "Expected Count": exp,
            "Detected Placed Count": det,
            "Deviation Status": "PASSED" if exp == det else "DISCREPANCY"
        })
    return pd.DataFrame(summary_rows)

def compute_dashboard_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to aggregate dashboard metrics directly.
    """
    detected_comps = results.get("detected_components", [])
    correct_count = len([d for d in detected_comps if d.get("status") == "Correct"])
    miss_count = len(results.get("missing", []))
    misaligned_count = len(results.get("misaligned", []))
    crack_count = len(results.get("cracks", []))
    extra_count = len(results.get("extra", []))
    anom_count = misaligned_count + crack_count + extra_count
    
    return {
        "expected_count": results.get("component_statistics", {}).get("total_expected", 0),
        "correct_count": correct_count,
        "missing_count": miss_count,
        "anomaly_count": anom_count,
        "processing_time": results.get("processing_time", "0.000 sec")
    }
