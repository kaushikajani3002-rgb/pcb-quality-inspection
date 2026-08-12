import time
import random
import math
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Tuple
from src.utils.constants import STATUS_PASS, STATUS_FAIL
from src.utils.logger import logger

class MockInspectionService:
    """
    Simulates high-fidelity PCB Automated Optical Inspection (AOI) results
    using template schemas. Controls PASS/FAIL status and coordinate mapping.
    """
    @staticmethod
    def run_inspection(
        template: Dict[str, Any],
        defect_mode: bool,
        confidence_threshold: float,
        iou_threshold: float,
        position_tolerance: float
    ) -> Dict[str, Any]:
        """
        Processes template components and runs a simulated comparison.
        Returns a structured dictionary matching the industrial AOI payload schema.
        """
        logger.info(f"Running mock inspection on template: {template.get('board_name', 'Unknown PCB')}")
        start_time = time.time()
        
        # Base templates metadata
        components_list = template.get("components", [])
        total_comps = len(components_list)
        
        board_dims = template.get("board_dimensions", {})
        width_mm = float(board_dims.get("width_mm", 100.0))
        height_mm = float(board_dims.get("height_mm", 100.0))

        # Count frequencies by type
        template_counts: Dict[str, int] = {}
        for c in components_list:
            t = c["type"]
            template_counts[t] = template_counts.get(t, 0) + 1

        # Detections list placeholders
        detected_comps: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []
        extra: List[Dict[str, Any]] = []
        misaligned: List[Dict[str, Any]] = []
        cracks: List[Dict[str, Any]] = []

        if not defect_mode:
            # PASSING CONDITION
            status = STATUS_PASS
            # Copy all template components directly as correct detections
            for idx, c in enumerate(components_list):
                detected_comps.append({
                    "id": c["id"],
                    "type": c["type"],
                    "center_x_pct": c["center_x_pct"],
                    "center_y_pct": c["center_y_pct"],
                    "width_pct": c["width_pct"],
                    "height_pct": c["height_pct"],
                    "confidence": round(random.uniform(0.88, 0.99), 2),
                    "status": "Correct"
                })
        else:
            # FAILING / ANOMALY CONDITION
            status = STATUS_FAIL
            
            # We will deliberately corrupt some components from the template:
            # 1. First component is marked MISSING
            # 2. Second component is marked MISALIGNED (shift center coordinates beyond tolerance)
            # 3. Add an EXTRA/rogue component not in the template
            # 4. Add a CRACK defect on one component's boundary
            
            for idx, c in enumerate(components_list):
                if idx == 0:
                    # Missing anomaly
                    missing.append({
                        "id": c["id"],
                        "type": c["type"],
                        "expected_x_pct": c["center_x_pct"],
                        "expected_y_pct": c["center_y_pct"],
                        "width_pct": c["width_pct"],
                        "height_pct": c["height_pct"],
                        "reason": "Absent from coordinate grid"
                    })
                elif idx == 1:
                    # Misaligned anomaly: Shift center coordinate beyond tolerance (in mm)
                    # We compute how many percentage units correspond to tolerance (mm) + extra offset
                    shift_mm = position_tolerance + 0.8  # shift beyond tolerance
                    shift_x_pct = shift_mm / width_mm
                    shift_y_pct = shift_mm / height_mm
                    
                    actual_x_pct = c["center_x_pct"] + shift_x_pct
                    actual_y_pct = c["center_y_pct"] + shift_y_pct
                    
                    dist_mm = round(math.sqrt((shift_x_pct * width_mm) ** 2 + (shift_y_pct * height_mm) ** 2), 2)
                    
                    misaligned.append({
                        "id": c["id"],
                        "type": c["type"],
                        "expected_x_pct": c["center_x_pct"],
                        "expected_y_pct": c["center_y_pct"],
                        "actual_x_pct": actual_x_pct,
                        "actual_y_pct": actual_y_pct,
                        "distance_mm": dist_mm,
                        "tolerance_mm": position_tolerance
                    })
                    
                    # Add to detected list but with actual displaced coordinates and status
                    detected_comps.append({
                        "id": c["id"],
                        "type": c["type"],
                        "center_x_pct": actual_x_pct,
                        "center_y_pct": actual_y_pct,
                        "width_pct": c["width_pct"],
                        "height_pct": c["height_pct"],
                        "confidence": round(random.uniform(0.75, 0.85), 2),
                        "status": "Misaligned"
                    })
                else:
                    # Remaining are correct
                    detected_comps.append({
                        "id": c["id"],
                        "type": c["type"],
                        "center_x_pct": c["center_x_pct"],
                        "center_y_pct": c["center_y_pct"],
                        "width_pct": c["width_pct"],
                        "height_pct": c["height_pct"],
                        "confidence": round(random.uniform(0.85, 0.98), 2),
                        "status": "Correct"
                    })

            # Add an extra component
            extra.append({
                "id": "ERR_EXTRA_01",
                "type": "Capacitor",
                "center_x_pct": 0.15,
                "center_y_pct": 0.15,
                "width_pct": 0.05,
                "height_pct": 0.05,
                "confidence": round(random.uniform(0.65, 0.72), 2),
                "reason": "Unregistered component at coordinate"
            })
            detected_comps.append({
                "id": "ERR_EXTRA_01",
                "type": "Capacitor",
                "center_x_pct": 0.15,
                "center_y_pct": 0.15,
                "width_pct": 0.05,
                "height_pct": 0.05,
                "confidence": round(random.uniform(0.65, 0.72), 2),
                "status": "Extra"
            })

            # Add a crack defect on one of the correct components (e.g., the last one)
            if len(detected_comps) > 2:
                # Find a non-extra component to crack
                crack_target = None
                for d in reversed(detected_comps):
                    if d["status"] == "Correct":
                        crack_target = d
                        break
                
                if crack_target:
                    cracks.append({
                        "id": f"CRK_{crack_target['id']}",
                        "parent_component": crack_target["id"],
                        "center_x_pct": crack_target["center_x_pct"] + (crack_target["width_pct"] / 3.0),
                        "center_y_pct": crack_target["center_y_pct"] + (crack_target["height_pct"] / 3.0),
                        "severity": "High (Solder Joint Fracture)"
                    })
                    # Update status of this component to Crack
                    crack_target["status"] = "Crack Detected"

        # Update final counts based on detections
        detected_counts: Dict[str, int] = {}
        for d in detected_comps:
            if d["status"] != "Extra":
                t = d["type"]
                detected_counts[t] = detected_counts.get(t, 0) + 1

        duration = time.time() - start_time
        processing_time_str = f"{duration:.3f} sec"

        return {
            "status": status,
            "processing_time": processing_time_str,
            "template_name": template.get("board_name", "Unknown PCB"),
            "component_statistics": {
                "total_expected": total_comps,
                "total_detected": len(detected_comps),
                "by_type": template_counts
            },
            "detected_components": detected_comps,
            "missing": missing,
            "extra": extra,
            "misaligned": misaligned,
            "cracks": cracks
        }

    @staticmethod
    def generate_mock_images(
        template: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Creates three Pillow images mimicking industrial visual analysis:
        1. Original RGB PCB image.
        2. Bounding Box & Label overlays (Inspection view).
        3. Heatmap / Segmentation overlays (Solder joint & crack view).
        """
        # Canvas dimensions from template
        dims = template.get("board_dimensions", {"pixel_width": 1000, "pixel_height": 700})
        w = int(dims.get("pixel_width", 1000))
        h = int(dims.get("pixel_height", 700))

        # --- 1. ORIGINAL IMAGE ---
        original_img = Image.new("RGB", (w, h), color="#1e293b")  # Dark Slate Blue PCB base
        draw_orig = ImageDraw.Draw(original_img)

        # Draw circuit traces (PCB copper lines mockup)
        for i in range(15):
            draw_orig.line([(random.randint(0, w), random.randint(0, h)), 
                            (random.randint(0, w), random.randint(0, h))], 
                           fill="#0f172a", width=3)
            
        # Draw physical components as dark grey boxes
        for c in template.get("components", []):
            cx = int(c["center_x_pct"] * w)
            cy = int(c["center_y_pct"] * h)
            cw = int(c["width_pct"] * w)
            ch = int(c["height_pct"] * h)
            left = cx - cw // 2
            top = cy - ch // 2
            right = cx + cw // 2
            bottom = cy + ch // 2
            draw_orig.rectangle([left, top, right, bottom], fill="#475569", outline="#64748b", width=2)
            # Component terminals / pins (visual styling)
            if cw > 40:
                draw_orig.rectangle([left - 5, top + 10, left, top + 20], fill="#94a3b8")
                draw_orig.rectangle([right, top + 10, right + 5, top + 20], fill="#94a3b8")

        # --- 2. DETECTION IMAGE ---
        detection_img = original_img.copy()
        draw_det = ImageDraw.Draw(detection_img)

        # Draw detections based on status
        for d in results.get("detected_components", []):
            cx = int(d["center_x_pct"] * w)
            cy = int(d["center_y_pct"] * h)
            cw = int(d["width_pct"] * w)
            ch = int(d["height_pct"] * h)
            left, top = cx - cw // 2, cy - ch // 2
            right, bottom = cx + cw // 2, cy + ch // 2
            
            # Select color based on component status
            status = d["status"]
            if status == "Correct":
                box_color = "#00FF66"  # Green
            elif status == "Misaligned":
                box_color = "#FFCC00"  # Yellow
            elif status == "Extra":
                box_color = "#E11D48"  # Pinkish-Red for Extra
            elif status == "Crack Detected":
                box_color = "#3399FF"  # Blue for crack
            else:
                box_color = "#FFFFFF"

            draw_det.rectangle([left, top, right, bottom], outline=box_color, width=3)
            draw_det.text((left + 5, top + 5), f"{d['id']} ({d['confidence']:.2f})", fill=box_color)

            # Draw misalignment offset vectors
            if status == "Misaligned":
                # Find expected position from template
                expected_x_pct = d["center_x_pct"]
                expected_y_pct = d["center_y_pct"]
                for m in results.get("misaligned", []):
                    if m["id"] == d["id"]:
                        expected_x_pct = m["expected_x_pct"]
                        expected_y_pct = m["expected_y_pct"]
                        break
                
                exp_x_px = int(expected_x_pct * w)
                exp_y_px = int(expected_y_pct * h)
                
                # Draw marker at expected center and vector line to actual center
                draw_det.ellipse([exp_x_px - 4, exp_y_px - 4, exp_x_px + 4, exp_y_px + 4], fill="#00FF66")
                draw_det.line([(exp_x_px, exp_y_px), (cx, cy)], fill="#FFCC00", width=2)

        # Draw Missing Component Outlines in dashed/solid Red
        for m in results.get("missing", []):
            cx = int(m["expected_x_pct"] * w)
            cy = int(m["expected_y_pct"] * h)
            cw = int(m["width_pct"] * w)
            ch = int(m["height_pct"] * h)
            left, top = cx - cw // 2, cy - ch // 2
            right, bottom = cx + cw // 2, cy + ch // 2
            
            draw_det.rectangle([left, top, right, bottom], outline="#FF3333", width=2)
            draw_det.line([(left, top), (right, bottom)], fill="#FF3333", width=1)
            draw_det.line([(left, bottom), (right, top)], fill="#FF3333", width=1)
            draw_det.text((left + 5, top + 5), f"MISSING: {m['id']}", fill="#FF3333")

        # --- 3. SEGMENTATION / CRACK IMAGE ---
        segmentation_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_seg = ImageDraw.Draw(segmentation_img)

        # Fill semi-transparent masks for all components
        for d in results.get("detected_components", []):
            cx = int(d["center_x_pct"] * w)
            cy = int(d["center_y_pct"] * h)
            cw = int(d["width_pct"] * w)
            ch = int(d["height_pct"] * h)
            left, top = cx - cw // 2, cy - ch // 2
            right, bottom = cx + cw // 2, cy + ch // 2
            
            if d["status"] == "Correct":
                fill_color = (0, 255, 102, 60)   # Green tint
            elif d["status"] == "Misaligned":
                fill_color = (255, 204, 0, 80)   # Yellow tint
            elif d["status"] == "Crack Detected":
                fill_color = (51, 153, 255, 80)  # Blue tint
            else:
                fill_color = (225, 29, 72, 80)   # Red/Pink tint
            
            draw_seg.rectangle([left, top, right, bottom], fill=fill_color)

        # Draw Blue crack overlay indicators
        for crk in results.get("cracks", []):
            ccx = int(crk["center_x_pct"] * w)
            ccy = int(crk["center_y_pct"] * h)
            # Draw crack shape
            draw_seg.polygon(
                [(ccx - 10, ccy - 5), (ccx + 5, ccy - 8), (ccx - 2, ccy + 10), (ccx + 12, ccy + 3), (ccx - 8, ccy + 5)],
                fill=(51, 153, 255, 255), outline=(255, 255, 255, 255)
            )
            draw_seg.text((ccx - 20, ccy - 20), "CRACK DETECTED", fill="#3399FF")

        # Composite segmentation mask on top of original image
        final_seg_img = Image.alpha_composite(original_img.convert("RGBA"), segmentation_img).convert("RGB")

        return original_img, detection_img, final_seg_img
