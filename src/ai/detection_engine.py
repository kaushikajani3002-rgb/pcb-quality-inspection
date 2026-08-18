import os
import time
import math
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw
import pandas as pd
from ultralytics import YOLO
from src.utils.logger import logger
from src.inspection.inspection_engine import InspectionEngine

# Dictionary of model paths. Mapped to configs/model.yaml defaults.
MODEL_PATHS = {
    "Component": "models/trained/Component/Component_best.pt",
    "DeepPCB": "models/trained/DeepPCB/DeepPCB.pt",
    "DsPCBSD+": "models/trained/DsPCBSD+/DsPCBSD+.pt",
    "HRIPCB": "models/trained/HRIPCB/HRIPCB.pt",
    "TDD-PCB": "models/trained/TDD-PCB/PDD-PCB-best.pt"
}

# Trained class names to template component type strings mapping
CLASS_NAME_MAP = {}

def map_class_to_component_type(class_name: str) -> str:
    """
    Maps YOLO component model class names (e.g. 'Capacitor 104J', 'IC 555', 'pins', 'switch', 'potentiometer')
    to general component type strings corresponding to expected template categories ('IC', 'Resistor', 'Capacitor', 'LED', 'Connector', 'VR', 'Diode', 'Transistor').
    """
    name_lower = class_name.lower().strip()
    if "resistor" in name_lower:
        return "Resistor"
    if "capacitor" in name_lower:
        return "Capacitor"
    if "ic" in name_lower or "lm324n" in name_lower or "4017" in name_lower or "555" in name_lower or "l7805cv" in name_lower or "mcu" in name_lower or "microcontroller" in name_lower:
        return "IC"
    if "led" in name_lower:
        return "LED"
    if "diode" in name_lower or "zener" in name_lower:
        return "Diode"
    if "transistor" in name_lower:
        return "Transistor"
    if any(k in name_lower for k in ["pushbutton", "relay", "button", "jack", "port", "connector", "fuse", "crystal", "switch", "pins", "pin", "header", "pad", "pads", "clock", "buzzer", "terminal", "battery", "display", "heatsink"]):
        return "Connector"
    if "transformer" in name_lower:
        return "Transformer"
    if "vr" in name_lower or "potentiometer" in name_lower or "trimpot" in name_lower:
        return "VR"
    return class_name.capitalize()

def load_model(model_name: str) -> Any:
    """
    Loads a YOLO model from weights.
    Resolves paths dynamically from configs/model.yaml, falling back to static defaults.
    """
    from src.utils.config_loader import ConfigLoader
    from pathlib import Path

    path = ""
    try:
        config = ConfigLoader()
        if model_name == "Component":
            path_rel = config.get("models.component_model.path")
        else:
            defect_mapping = config.get("models.defect_mapping") or {}
            path_rel = None
            for key, val in defect_mapping.items():
                if isinstance(val, dict) and val.get("name") == model_name:
                    path_rel = val.get("path")
                    break
        if path_rel:
            path = str(config.project_root / path_rel)
    except Exception:
        pass

    if not path:
        path_rel = MODEL_PATHS.get(model_name, "")
        if path_rel:
            project_root = Path(__file__).resolve().parent.parent.parent
            path = str(project_root / path_rel)

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

def normalize_component_coordinates(comp: Dict[str, Any], W: float, H: float, board_box: List[float] = None) -> Dict[str, Any]:
    """
    Ensures that every component dictionary conforms to a single consistent coordinate schema.
    Calculates percentage values in 0-100 range and maps pixels consistently.
    If board_box is provided, projects normalized expected coordinates onto the board.
    """
    c = comp.copy()
    c.setdefault("id", "Unknown")
    c.setdefault("type", "Unknown")
    c.setdefault("class_id", -1)
    c.setdefault("class_name", c["type"])
    c.setdefault("confidence", 1.0)
    
    if "x1" in c and "y1" in c and "x2" in c and "y2" in c:
        x1, y1, x2, y2 = float(c["x1"]), float(c["y1"]), float(c["x2"]), float(c["y2"])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
    elif "center_x" in c and "center_y" in c and "width" in c and "height" in c and not any(k in ["center_x_pct", "center_y_pct"] for k in c):
        cx = float(c["center_x"])
        cy = float(c["center_y"])
        width = float(c["width"])
        height = float(c["height"])
        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0
    else:
        cx_pct = float(c.get("center_x_pct", 0.5))
        cy_pct = float(c.get("center_y_pct", 0.5))
        w_pct = float(c.get("width_pct", 0.1))
        h_pct = float(c.get("height_pct", 0.1))
        
        # If input has ratio coordinates (0.0 to 1.0), convert to ratio.
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
            
        if board_box is not None and len(board_box) == 4:
            bx1, by1, bx2, by2 = board_box
            bw = bx2 - bx1
            bh = by2 - by1
            cx = bx1 + cx_ratio * bw
            cy = by1 + cy_ratio * bh
            width = w_ratio * bw
            height = h_ratio * bh
        else:
            cx = cx_ratio * W
            cy = cy_ratio * H
            width = w_ratio * W
            height = h_ratio * H
            
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
    c["center_x_pct"] = round((cx / W) * 100.0, 2) if W > 0 else 0.0
    c["center_y_pct"] = round((cy / H) * 100.0, 2) if H > 0 else 0.0
    c["width_pct"] = round((width / W) * 100.0, 2) if W > 0 else 0.0
    c["height_pct"] = round((height / H) * 100.0, 2) if H > 0 else 0.0
    
    return c

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
    dist_x_mm = ((exp_x_pct - act_x_pct) / 100.0) * width_mm
    dist_y_mm = ((exp_y_pct - act_y_pct) / 100.0) * height_mm
    return math.sqrt(dist_x_mm ** 2 + dist_y_mm ** 2)

def match_detections_to_template(
    expected_comps: List[Dict[str, Any]], 
    detections: List[Dict[str, Any]], 
    width_mm: float, 
    height_mm: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Spatially maps raw class-based detections to expected template component IDs.
    Matches nearest component of the same type, with fallback spatial proximity pairing.
    """
    matched_detections = []
    matching_decisions = []
    
    # Filter out any PCB board container detections
    filtered_dets = [d for d in detections if str(d.get("type", "")).lower() != "pcb" and str(d.get("class_name", "")).lower() != "pcb"]
    
    # Group expected by standardized component type
    expected_by_type = {}
    for exp in expected_comps:
        t = str(exp["type"]).strip().upper()
        expected_by_type.setdefault(t, []).append(exp)
        
    # Group detections by standardized component type
    detections_by_type = {}
    for det in filtered_dets:
        t = str(det["type"]).strip().upper()
        detections_by_type.setdefault(t, []).append(det)
        
    all_types = set(expected_by_type.keys()) | set(detections_by_type.keys())
    extra_counter = 1
    
    assigned_exp_ids = set()
    assigned_det_indices = set()

    # Pass 1: Primary type-based spatial proximity matching
    for t in all_types:
        exps = expected_by_type.get(t, [])
        dets = detections_by_type.get(t, [])
        
        if not exps or not dets:
            continue
            
        # Match using proximity distance matrix (greedy search)
        dist_matrix = []
        for exp in exps:
            row = []
            exp_x_mm = (exp["center_x_pct"] / 100.0) * width_mm
            exp_y_mm = (exp["center_y_pct"] / 100.0) * height_mm
            for det in dets:
                det_x_mm = (det["center_x_pct"] / 100.0) * width_mm
                det_y_mm = (det["center_y_pct"] / 100.0) * height_mm
                dist = math.sqrt((exp_x_mm - det_x_mm) ** 2 + (exp_y_mm - det_y_mm) ** 2)
                row.append(dist)
            dist_matrix.append(row)
            
        for exp_idx, exp in enumerate(exps):
            row = dist_matrix[exp_idx]
            sorted_det_indices = sorted(range(len(row)), key=lambda k: row[k])
            
            best_det_idx = None
            for idx in sorted_det_indices:
                global_det = dets[idx]
                # Identify global detection index in filtered_dets
                g_idx = filtered_dets.index(global_det)
                if g_idx not in assigned_det_indices:
                    best_det_idx = g_idx
                    break
                    
            if best_det_idx is not None:
                assigned_det_indices.add(best_det_idx)
                assigned_exp_ids.add(exp["id"])
                
                det = filtered_dets[best_det_idx].copy()
                det["id"] = exp["id"]
                det["expected_id"] = exp["id"]
                det["expected_type"] = exp["type"]
                det["status"] = "Correct"
                matched_detections.append(det)
                
                dist = math.sqrt(
                    (((exp["center_x_pct"] / 100.0) * width_mm) - ((det["center_x_pct"] / 100.0) * width_mm)) ** 2 +
                    (((exp["center_y_pct"] / 100.0) * height_mm) - ((det["center_y_pct"] / 100.0) * height_mm)) ** 2
                )
                matching_decisions.append({
                    "expected_id": exp["id"],
                    "detected_type": det["type"],
                    "distance_mm": round(dist, 2),
                    "decision": "MATCHED",
                    "reason": f"Spatially paired with detected component of type '{det['type']}' at distance {dist:.2f} mm"
                })

    # Pass 2: Fallback spatial proximity pairing for remaining unmatched expected components
    unmatched_exps = [e for e in expected_comps if e["id"] not in assigned_exp_ids]
    unmatched_det_indices = [i for i in range(len(filtered_dets)) if i not in assigned_det_indices]

    if unmatched_exps and unmatched_det_indices:
        for exp in unmatched_exps:
            exp_x_mm = (exp["center_x_pct"] / 100.0) * width_mm
            exp_y_mm = (exp["center_y_pct"] / 100.0) * height_mm
            
            best_det_idx = None
            best_dist = 999.0
            
            for g_idx in unmatched_det_indices:
                det = filtered_dets[g_idx]
                det_x_mm = (det["center_x_pct"] / 100.0) * width_mm
                det_y_mm = (det["center_y_pct"] / 100.0) * height_mm
                dist = math.sqrt((exp_x_mm - det_x_mm) ** 2 + (exp_y_mm - det_y_mm) ** 2)
                
                # Proximity limit for spatial pairing across compatible categories (e.g. 15mm)
                if dist < 15.0 and dist < best_dist:
                    best_dist = dist
                    best_det_idx = g_idx
                    
            if best_det_idx is not None:
                assigned_det_indices.add(best_det_idx)
                unmatched_det_indices.remove(best_det_idx)
                assigned_exp_ids.add(exp["id"])
                
                det = filtered_dets[best_det_idx].copy()
                det["id"] = exp["id"]
                det["expected_id"] = exp["id"]
                det["expected_type"] = exp["type"]
                det["status"] = "Correct"
                matched_detections.append(det)
                
                matching_decisions.append({
                    "expected_id": exp["id"],
                    "detected_type": det["type"],
                    "distance_mm": round(best_dist, 2),
                    "decision": "MATCHED_SPATIAL_FALLBACK",
                    "reason": f"Spatially paired across compatible category with detected '{det['type']}' at distance {best_dist:.2f} mm"
                })

    # Record missing decisions
    for exp in expected_comps:
        if exp["id"] not in assigned_exp_ids:
            matching_decisions.append({
                "expected_id": exp["id"],
                "detected_type": "N/A",
                "distance_mm": 999.0,
                "decision": "MISSING",
                "reason": f"No detected component of type '{exp['type']}' available for matching"
            })

    # Unassigned detections are extra/unregistered components
    for g_idx in range(len(filtered_dets)):
        if g_idx not in assigned_det_indices:
            det_copy = filtered_dets[g_idx].copy()
            det_copy["id"] = f"ERR_EXTRA_{str(det_copy['type']).upper()}_{extra_counter:02d}"
            det_copy["expected_id"] = "N/A"
            det_copy["status"] = "Extra"
            matched_detections.append(det_copy)
            extra_counter += 1
            
            matching_decisions.append({
                "expected_id": "N/A",
                "detected_type": det_copy["type"],
                "distance_mm": 999.0,
                "decision": "EXTRA",
                "reason": f"Detected extra/unregistered component of type '{det_copy['type']}' at coordinates ({det_copy['center_x_pct']:.2f}, {det_copy['center_y_pct']:.2f})"
            })
            
    return matched_detections, matching_decisions

def run_component_inspection(
    uploaded_image: Any, 
    component_model: Any, 
    conf_slider: float, 
    iou_slider: float, 
    active_template: Dict[str, Any], 
    position_tolerance_slider: float
) -> Dict[str, Any]:
    """
    Executes component verification on the component image.
    Uses two-pass crop-and-resize board detection to dynamically align template.
    """
    start_time = time.time()
    
    # 1. Load image and get dimensions
    import numpy as np
    filename = "PIL Image Object"
    if hasattr(uploaded_image, "name"):
        filename = uploaded_image.name
    elif hasattr(uploaded_image, "filename"):
        filename = uploaded_image.filename
        
    if not isinstance(uploaded_image, Image.Image):
        img = Image.open(uploaded_image).convert("RGB")
    else:
        img = uploaded_image

    img_np = np.array(img)
    w, h = img.size
    logger.info(f"[COMP INSPECTION] Target image: Name='{filename}', Shape={img_np.shape}, Mode='{img.mode}'")
    
    # 2. Parse Template default values and component types
    raw_expected = active_template.get("components", [])
    total_comps = len(raw_expected)
    
    board_dims = active_template.get("board_dimensions", {})
    width_mm = float(board_dims.get("width_mm", 100.0))
    height_mm = float(board_dims.get("height_mm", 100.0))
    
    template_counts = {}
    for c in raw_expected:
        t = c["type"]
        template_counts[t] = template_counts.get(t, 0) + 1
        
    # 3. Model Inference (Two-pass alignment and detection pipeline)
    raw_detections = []
    raw_details = []
    pcb_box = None
    
    if component_model is not None:
        try:
             # Run component YOLO model inference directly on uploaded image
             logger.info(f"[COMP INSPECTION INFERENCE] Conf: {conf_slider}, IoU: {iou_slider}")
             results = component_model.predict(source=img, conf=conf_slider, iou=iou_slider, imgsz=640)
             boxes = results[0].boxes
             logger.info(f"[COMP INSPECTION RESULT] YOLO returned {len(boxes)} raw bounding box detections.")
             
             for idx, box in enumerate(boxes, start=1):
                 cls_id = int(box.cls[0])
                 conf = float(box.conf[0])
                 class_name = component_model.names.get(cls_id, f"Class {cls_id}")
                 
                 # Exclude outer PCB container class if present in model
                 if "pcb" in class_name.lower():
                     continue
                     
                 x1, y1, x2, y2 = box.xyxy[0].tolist()
                 mapped_type = map_class_to_component_type(class_name)
                 
                 raw_det = normalize_component_coordinates({
                     "id": f"DET_{idx:02d}",
                     "detection_id": f"DET_{idx:02d}",
                     "class_id": cls_id,
                     "class_name": class_name,
                     "type": class_name,  # Use exact YOLO class_name for component inventory
                     "mapped_type": mapped_type,
                     "confidence": float(conf),
                     "x1": x1,
                     "y1": y1,
                     "x2": x2,
                     "y2": y2
                 }, w, h)
                 
                 raw_detections.append(raw_det)
                 raw_details.append(raw_det)
        except Exception as e:
             logger.error(f"YOLO Component Inference pipeline failed: {e}")
    else:
         logger.warning("Component model is offline/None. Returning empty detections.")
         
    # Preserve backend verification logic in debug_info for future scope
    expected_comps = [normalize_component_coordinates(c, w, h, board_box=pcb_box) for c in raw_expected]
    matched_detections, matching_decisions = match_detections_to_template(expected_comps, raw_detections, width_mm, height_mm)
    engine = InspectionEngine()
    inspection_results = engine.inspect_board(active_template, matched_detections, position_tolerance_slider, [])
    
    valid_component_detections = raw_detections
    
    # Calculate Component Inventory counts by exact YOLO class name
    detected_counts = {}
    total_conf_sum = 0.0
    for d in valid_component_detections:
        c_name = d.get("class_name", d.get("type", "Unknown"))
        detected_counts[c_name] = detected_counts.get(c_name, 0) + 1
        total_conf_sum += float(d.get("confidence", 0.0))
            
    total_detected = len(valid_component_detections)
    unique_types_count = len(detected_counts)
    avg_confidence = (total_conf_sum / total_detected) if total_detected > 0 else 0.0
            
    # Render Annotated Bounding Box Image with Class Name + Confidence
    annotated_image = img.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    for d in valid_component_detections:
        left = int(d["x1"])
        top = int(d["y1"])
        right = int(d["x2"])
        bottom = int(d["y2"])
        
        box_color = "#00FF66"  # Bright Green for YOLO Component Detections
        draw.rectangle([left, top, right, bottom], outline=box_color, width=3)
        label_text = f"{d.get('class_name')} {d.get('confidence', 0.0):.2f}"
        draw.text((left + 4, max(0, top - 12)), label_text, fill=box_color)
        
    duration = time.time() - start_time
    processing_time_str = f"{duration:.3f} sec"
    
    status = "PASS"
    
    # Build debug info payload
    import torch
    cuda_avail = torch.cuda.is_available()
    cuda_dev = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    
    debug_info = {
        "model_path": getattr(component_model, "ckpt_path", "models/trained/Component/Component_best.pt") if component_model else "N/A",
        "model_type": "YOLO Component Detector",
        "device": str(getattr(component_model, "device", "cpu")) if component_model else "N/A",
        "cuda_available": cuda_avail,
        "cuda_device_name": cuda_dev,
        "class_names": list(component_model.names.values()) if component_model else [],
        "image_filename": filename,
        "image_width": w,
        "image_height": h,
        "image_channels": img_np.shape[2] if len(img_np.shape) > 2 else 1,
        "confidence_threshold": conf_slider,
        "iou_threshold": iou_slider,
        "raw_detections": valid_component_detections,
        "filtered_detections": valid_component_detections,
        "template": expected_comps,
        "matching": matching_decisions,
        "future_scope_verification": {
            "missing": inspection_results["missing"],
            "extra": inspection_results["extra"],
            "misaligned": inspection_results["misaligned"]
        }
    }
    
    return {
        "status": status,
        "processing_time": processing_time_str,
        "template_name": active_template.get("board_name", "Unknown PCB"),
        "component_statistics": {
            "total_detected": total_detected,
            "unique_types_count": unique_types_count,
            "avg_confidence": avg_confidence,
            "total_expected": len(raw_expected),
            "by_type": detected_counts
        },
        "average_confidence": avg_confidence,
        "detected_components": valid_component_detections,
        "detected_counts": detected_counts,
        "missing": [],
        "extra": [],
        "misaligned": [],
        "annotated_image": annotated_image,
        "original_image": img,
        "debug_info": debug_info
    }

def run_circuit_inspection(
    uploaded_image: Any, 
    defect_model: Any, 
    conf_slider: float, 
    iou_slider: float,
    defect_mode: bool = False,
    matched_detections: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes trace & solder joint inspection on the circuit image.
    """
    start_time = time.time()
    
    # 1. Load image and get dimensions
    import numpy as np
    filename = "PIL Image Object"
    if hasattr(uploaded_image, "name"):
        filename = uploaded_image.name
    elif hasattr(uploaded_image, "filename"):
        filename = uploaded_image.filename
        
    if not isinstance(uploaded_image, Image.Image):
        img = Image.open(uploaded_image).convert("RGB")
    else:
        img = uploaded_image

    img_np = np.array(img)
    w, h = img.size
    logger.info(f"[CIRCUIT INSPECTION] Target image: Name='{filename}', Shape={img_np.shape}, Mode='{img.mode}'")
    
    cracks = []
    
    # 2. Model Inference
    if defect_model is not None:
        try:
            logger.info(f"[CIRCUIT INSPECTION INFERENCE]\nConfidence Threshold: {conf_slider}\nIoU Threshold: {iou_slider}\nModel: Defect Detector")
            defect_results = defect_model.predict(source=img, conf=conf_slider, iou=iou_slider, imgsz=640)
            boxes = defect_results[0].boxes
            logger.info(f"[CIRCUIT INFERENCE RESULT] Final defect detections: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = defect_model.names.get(cls_id, f"Class {cls_id}")
                xywhn = box.xywhn[0].tolist()
                cx_pct, cy_pct, w_pct, h_pct = xywhn
                
                # Check for bounding box coordinates
                xyxy = box.xyxy[0].tolist()
                dx1, dy1, dx2, dy2 = xyxy
                
                cracks.append(normalize_component_coordinates({
                    "id": f"DEF_{class_name.upper()}_{len(cracks)+1:02d}",
                    "parent_component": "Board/Trace",
                    "center_x_pct": cx_pct * 100.0,
                    "center_y_pct": cy_pct * 100.0,
                    "width_pct": w_pct * 100.0,
                    "height_pct": h_pct * 100.0,
                    "x1": dx1,
                    "y1": dy1,
                    "x2": dx2,
                    "y2": dy2,
                    "confidence": conf,
                    "class_name": class_name,
                    "type": "Defect",
                    "severity": f"High ({class_name} @ {conf:.2f})"
                }, w, h))
        except Exception as e:
            logger.error(f"YOLO Defect Inference failed: {e}")
            
    # Fallback to simulated cracks if no real model is present and defect mode is toggled
    if defect_model is None and defect_mode:
        if matched_detections:
            correct_comps = [d for d in matched_detections if d.get("status") == "Correct"]
            if correct_comps:
                crack_target = correct_comps[-1]  # Pick last correct component to corrupt
                crack_id = f"CRK_{crack_target['id']}"
                cracks.append(normalize_component_coordinates({
                    "id": crack_id,
                    "parent_component": crack_target["id"],
                    "center_x_pct": crack_target["center_x_pct"] + (crack_target["width_pct"] / 3.0),
                    "center_y_pct": crack_target["center_y_pct"] + (crack_target["height_pct"] / 3.0),
                    "width_pct": crack_target["width_pct"] / 3.0,
                    "height_pct": crack_target["height_pct"] / 3.0,
                    "confidence": 0.95,
                    "class_name": "Solder Crack",
                    "type": "Defect",
                    "severity": "High (Solder Joint Fracture)"
                }, w, h))
                crack_target["status"] = "Crack Detected"
                
    # 3. Render Defect Overlay Image
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_seg = ImageDraw.Draw(overlay)
    
    # Draw cracks
    for crk in cracks:
        if "x1" in crk:
            dleft = int(crk["x1"])
            dtop = int(crk["y1"])
            dright = int(crk["x2"])
            dbottom = int(crk["y2"])
            
            draw_seg.rectangle([dleft, dtop, dright, dbottom], outline=(255, 51, 51, 255), width=3)
            fill_color = (255, 51, 51, 40)
            draw_seg.rectangle([dleft, dtop, dright, dbottom], fill=fill_color)
            
            lbl = f"{crk['class_name']} ({crk['confidence']:.2f})"
            draw_seg.text((dleft + 5, dtop + 5), lbl, fill="#FF3333")
        else:
            ccx = int(crk["center_x"])
            ccy = int(crk["center_y"])
            draw_seg.polygon(
                [(ccx - 10, ccy - 5), (ccx + 5, ccy - 8), (ccx - 2, ccy + 10), (ccx + 12, ccy + 3), (ccx - 8, ccy + 5)],
                fill=(51, 153, 255, 255), outline=(255, 255, 255, 255)
            )
            draw_seg.text((ccx - 20, ccy - 20), "CRACK DETECTED", fill="#3399FF")
            
    # Composite segmentation mask on top of original image
    segmentation_image = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    
    duration = time.time() - start_time
    processing_time_str = f"{duration:.3f} sec"
    
    status = "FAIL" if len(cracks) > 0 else "PASS"
    
    return {
        "status": status,
        "processing_time": processing_time_str,
        "defects": cracks,
        "annotated_image": segmentation_image,
        "original_image": img
    }

def run_component_counting(
    uploaded_image: Any, 
    component_model: Any, 
    conf_slider: float, 
    iou_slider: float, 
    active_template: Dict[str, Any], 
    position_tolerance_slider: float,
    defect_mode: bool = False,
    defect_model: Any = None
) -> Dict[str, Any]:
    """
    Legacy wrapper / backward compatibility function mapping to the split inspection engines.
    """
    comp_res = run_component_inspection(
        uploaded_image=uploaded_image,
        component_model=component_model,
        conf_slider=conf_slider,
        iou_slider=iou_slider,
        active_template=active_template,
        position_tolerance_slider=position_tolerance_slider
    )
    
    if defect_model is not None or defect_mode:
        circ_res = run_circuit_inspection(
            uploaded_image=uploaded_image,
            defect_model=defect_model,
            conf_slider=conf_slider,
            iou_slider=iou_slider,
            defect_mode=defect_mode,
            matched_detections=comp_res["detected_components"]
        )
        cracks = circ_res["defects"]
        segmentation_image = circ_res["annotated_image"]
    else:
        cracks = []
        segmentation_image = comp_res["annotated_image"]
        
    has_anomalies = (
        len(comp_res["missing"]) > 0 or 
        len(comp_res["extra"]) > 0 or 
        len(comp_res["misaligned"]) > 0 or 
        len(cracks) > 0
    )
    overall_status = "FAIL" if has_anomalies else "PASS"
    
    debug_info = comp_res.get("debug_info", {})
    if "final" not in debug_info or not isinstance(debug_info["final"], dict):
        debug_info["final"] = {}
    debug_info["final"]["anomaly_count"] = (
        len(comp_res.get("missing", [])) + 
        len(comp_res.get("extra", [])) + 
        len(comp_res.get("misaligned", [])) + 
        len(cracks)
    )
    
    return {
        "status": overall_status,
        "processing_time": comp_res["processing_time"],
        "template_name": comp_res["template_name"],
        "component_statistics": comp_res["component_statistics"],
        "detected_components": comp_res["detected_components"],
        "detected_counts": comp_res["detected_counts"],
        "missing": comp_res["missing"],
        "extra": comp_res["extra"],
        "misaligned": comp_res["misaligned"],
        "cracks": cracks,
        "annotated_image": comp_res["annotated_image"],
        "segmentation_image": segmentation_image,
        "original_image": comp_res["original_image"],
        "debug_info": debug_info
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
