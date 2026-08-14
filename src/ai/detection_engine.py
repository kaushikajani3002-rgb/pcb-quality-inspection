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
    "Component": "models/trained/Component/All_cercit_finetuned_best.pt",
    "DeepPCB": "models/trained/DeepPCB/DeepPCB.pt",
    "DsPCBSD+": "models/trained/DsPCBSD+/DsPCBSD+.pt",
    "HRIPCB": "models/trained/HRIPCB/HRIPCB.pt",
    "TDD-PCB": "models/trained/TDD-PCB/PDD-PCB-best.pt"
}

# Trained class names to template component type strings mapping
CLASS_NAME_MAP = {}

def map_class_to_component_type(class_name: str) -> str:
    """
    Maps 58 YOLO component model class names (e.g. 'Capacitor 104J', 'IC 555')
    to general component type strings corresponding to expected template categories ('IC', 'Resistor', 'Capacitor', 'LED', 'Connector').
    """
    name_lower = class_name.lower()
    if "resistor" in name_lower:
        return "Resistor"
    if "capacitor" in name_lower:
        return "Capacitor"
    if "ic" in name_lower or "lm324n" in name_lower or "4017" in name_lower or "555" in name_lower or "l7805cv" in name_lower:
        return "IC"
    if "led" in name_lower:
        return "LED"
    if "diode" in name_lower:
        return "Diode"
    if "transistor" in name_lower:
        return "Transistor"
    if "pushbutton" in name_lower or "relay" in name_lower or "button" in name_lower or "jack" in name_lower or "port" in name_lower or "connector" in name_lower or "fuse" in name_lower or "crystal" in name_lower:
        return "Connector"
    if "transformer" in name_lower:
        return "Transformer"
    if "vr" in name_lower:
        return "VR"
    return class_name

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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Spatially maps raw class-based detections to expected template component IDs.
    Matches nearest component of the same type.
    """
    matched_detections = []
    matching_decisions = []
    
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
                
                # Record matching decision for extra
                matching_decisions.append({
                    "expected_id": "N/A",
                    "detected_type": det["type"],
                    "distance_mm": 999.0,
                    "decision": "EXTRA",
                    "reason": f"Detected extra/unregistered component of type '{det['type']}'"
                })
            continue
            
        if not dets:
            # All expected of this type are missing (handled downstream by MissingChecker)
            for exp in exps:
                matching_decisions.append({
                    "expected_id": exp["id"],
                    "detected_type": "N/A",
                    "distance_mm": 999.0,
                    "decision": "MISSING",
                    "reason": f"No detected component of type '{t}' available for matching"
                })
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
                
                # Record matching decision
                dist = row[best_det_idx]
                matching_decisions.append({
                    "expected_id": exp["id"],
                    "detected_type": det["type"],
                    "distance_mm": round(dist, 2),
                    "decision": "MATCHED",
                    "reason": f"Spatially paired with detected component of type '{det['type']}' at distance {dist:.2f} mm"
                })
            else:
                # Expected component was not matched to any detection
                matching_decisions.append({
                    "expected_id": exp["id"],
                    "detected_type": "N/A",
                    "distance_mm": 999.0,
                    "decision": "MISSING",
                    "reason": f"No detected component of type '{t}' available for matching"
                })
                
        # Unassigned detections are extra/unregistered
        for det_idx, det in enumerate(dets):
            if det_idx not in assigned_dets:
                det_copy = det.copy()
                det_copy["id"] = f"ERR_EXTRA_{t.upper()}_{extra_counter:02d}"
                det_copy["status"] = "Extra"
                matched_detections.append(det_copy)
                extra_counter += 1
                
                # Record matching decision for extra
                matching_decisions.append({
                    "expected_id": "N/A",
                    "detected_type": det["type"],
                    "distance_mm": 999.0,
                    "decision": "EXTRA",
                    "reason": f"Detected extra/unregistered component of type '{det['type']}' at coordinates ({det['center_x_pct']:.2f}, {det['center_y_pct']:.2f})"
                })
                
    return matched_detections, matching_decisions

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
    Executes YOLO11m component detection on an uploaded image, maps the predictions 
    to reference template coordinates, runs the verification engine, and renders the overlays.
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
    logger.info(f"[IMAGE UPLOAD] Received target image: Name='{filename}', Shape={img_np.shape}, Mode='{img.mode}', Dtype={img_np.dtype}")

    # Apply PCB perspective alignment warping
    try:
        from src.ai.alignment import align_pcb_image
        from src.utils.config_loader import ConfigLoader
        # Check if reference image is available (e.g. from template)
        ref_image_path = active_template.get("reference_image")
        ref_img = None
        if ref_image_path:
            config_loader = ConfigLoader()
            ref_path = config_loader.project_root / ref_image_path
            if ref_path.exists():
                ref_img = Image.open(ref_path).convert("RGB")
                
        if ref_img is not None:
            logger.info("Applying ORB homography alignment warping on target image...")
            img = align_pcb_image(img, ref_img)
    except Exception as e:
        logger.error(f"PCB Image Alignment warping failed: {e}. Proceeding with original target image.")

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
    raw_details = []
    if component_model is not None:
        try:
            logger.info(f"[INFERENCE]\nConfidence Threshold: {conf_slider}\nIoU Threshold: {iou_slider}\nModel: Component Detector")
            results = component_model.predict(source=img, conf=conf_slider, iou=iou_slider, imgsz=640)
            boxes = results[0].boxes
            logger.info(f"[INFERENCE RESULT] Final component detections after confidence & NMS: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = component_model.names.get(cls_id, f"Class {cls_id}")
                
                # Extract pixel coordinates
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                xywh = box.xywh[0].tolist()
                cx, cy, w_box, h_box = xywh
                
                logger.info(f"  [RAW DET] ClassID: {cls_id}, Name: '{class_name}', Conf: {conf:.4f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Center: ({cx:.1f}, {cy:.1f}), Size: {w_box:.1f}x{h_box:.1f}")
                
                # Apply class mapping
                mapped_type = map_class_to_component_type(class_name)
                
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
                
                raw_details.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": cx,
                    "center_y": cy,
                    "width": w_box,
                    "height": h_box
                })
        except Exception as e:
            logger.error(f"YOLO Component Inference failed: {e}")
    else:
        logger.warning("Component model is offline/None. Returning empty detections.")
        
    # 4. Spatially match detections to template expected layout
    matched_detections, matching_decisions = match_detections_to_template(expected_comps, raw_detections, width_mm, height_mm)
    
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
            
    # 6. Solder & Trace Defect Detection (Real model inference or Simulation fallback)
    cracks = []
    if defect_model is not None:
        try:
            logger.info(f"[INFERENCE]\nConfidence Threshold: {conf_slider}\nIoU Threshold: {iou_slider}\nModel: Defect Detector")
            defect_results = defect_model.predict(source=img, conf=conf_slider, iou=iou_slider, imgsz=640)
            boxes = defect_results[0].boxes
            logger.info(f"[INFERENCE RESULT] Final defect detections after confidence & NMS: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = defect_model.names.get(cls_id, f"Class {cls_id}")
                xywhn = box.xywhn[0].tolist()
                cx_pct, cy_pct, w_pct, h_pct = xywhn
                
                cracks.append({
                    "id": f"DEF_{class_name.upper()}_{len(cracks)+1:02d}",
                    "parent_component": "Board/Trace",
                    "center_x_pct": cx_pct,
                    "center_y_pct": cy_pct,
                    "width_pct": w_pct,
                    "height_pct": h_pct,
                    "severity": f"High ({class_name} @ {conf:.2f})"
                })
        except Exception as e:
            logger.error(f"YOLO Defect Inference failed: {e}")
            
    # Fallback to simulation cracks if no real model is present and defect mode is toggled
    if defect_model is None and defect_mode:
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
        
    # Draw crack and trace defect shapes
    for crk in cracks:
        if "width_pct" in crk:
            dcx = int(crk["center_x_pct"] * w)
            dcy = int(crk["center_y_pct"] * h)
            dcw = int(crk["width_pct"] * w)
            dch = int(crk["height_pct"] * h)
            dleft, dtop = dcx - dcw // 2, dcy - dch // 2
            dright, dbottom = dcx + dcw // 2, dcy + dch // 2
            # Draw red defect bounding box
            draw_seg.rectangle([dleft, dtop, dright, dbottom], outline=(255, 51, 51, 255), width=3)
            # Write label
            lbl = crk["severity"].split(" ")[-3].replace("(", "")
            draw_seg.text((dleft + 5, dtop + 5), lbl, fill="#FF3333")
        else:
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
    
    # Build dynamic debug info payload for Streamlit operator dashboard
    import torch
    cuda_avail = torch.cuda.is_available()
    cuda_dev = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    
    debug_info = {
        "model_path": getattr(component_model, "ckpt_path", "models/trained/Component/All_cercit_finetuned_best.pt") if component_model else "N/A",
        "model_type": "YOLO Object Detector",
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
        "raw_detections": raw_details,
        "filtered_detections": [
            {
                "id": d.get("id", "N/A"),
                "type": d.get("type", "Unknown"),
                "center_x_pct": d.get("center_x_pct", 0.0),
                "center_y_pct": d.get("center_y_pct", 0.0),
                "status": d.get("status", "Unknown"),
                "confidence": d.get("confidence", 0.0)
            } for d in matched_detections
        ],
        "template": [
            {
                "id": c.get("id", "N/A"),
                "type": c.get("type", "Unknown"),
                "center_x_pct": c.get("center_x_pct", 0.0),
                "center_y_pct": c.get("center_y_pct", 0.0)
            } for c in expected_comps
        ],
        "matching": matching_decisions,
        "final": {
            "detected_count": len(matched_detections),
            "placed_count": len([d for d in matched_detections if d.get("status") in ["Correct", "Misaligned"]]),
            "missing_count": len(inspection_results["missing"]),
            "extra_count": len(inspection_results["extra"]),
            "anomaly_count": len(inspection_results["missing"]) + len(inspection_results["extra"]) + len(inspection_results["misaligned"]) + len(cracks)
        }
    }
    
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
        "original_image": img,
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
