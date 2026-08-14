import math
from typing import Dict, Any, List

class PositionChecker:
    """
    Checks component alignment. Compares detected centers with expected 
    template coordinates in millimeters using Euclidean Distance on normalized coordinates.
    """
    def __init__(self):
        pass

    def check(
        self, 
        expected: List[Dict[str, Any]], 
        detected: List[Dict[str, Any]], 
        default_tolerance_mm: float,
        board_dimensions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Calculates distance between matching component IDs in millimeters. 
        If distance exceeds tolerance (mm), it is marked as misaligned.
        
        :param expected: List of expected template components (normalized coordinates).
        :param detected: List of detected components (normalized coordinates).
        :param default_tolerance_mm: Tolerance threshold in millimeters.
        :param board_dimensions: Dict containing physical width_mm and height_mm.
        :return: List of misaligned component information dictionaries.
        """
        misaligned_components: List[Dict[str, Any]] = []
        expected_map = {e["id"]: e for e in expected if "id" in e}
        
        # Get board physical dimensions, fallback to 1.0 to avoid division/zero errors
        width_mm = float(board_dimensions.get("width_mm", 100.0))
        height_mm = float(board_dimensions.get("height_mm", 100.0))
        
        for d in detected:
            d_id = d.get("id")
            if d_id in expected_map:
                exp_comp = expected_map[d_id]
                
                # Check for tolerance override (in mm) in the template itself, otherwise use default
                tolerance = float(exp_comp.get("tolerance_override_mm", default_tolerance_mm))
                
                # Retrieve expected and actual normalized coordinates
                exp_x_pct = float(exp_comp.get("center_x_pct", 0.0))
                exp_y_pct = float(exp_comp.get("center_y_pct", 0.0))
                act_x_pct = float(d.get("center_x_pct", 0.0))
                act_y_pct = float(d.get("center_y_pct", 0.0))
                
                # Convert normalized percentages (0-100) to physical millimeters
                exp_x_mm = (exp_x_pct / 100.0) * width_mm
                exp_y_mm = (exp_y_pct / 100.0) * height_mm
                act_x_mm = (act_x_pct / 100.0) * width_mm
                act_y_mm = (act_y_pct / 100.0) * height_mm
                
                # Euclidean distance in millimeters
                distance = math.sqrt((exp_x_mm - act_x_mm) ** 2 + (exp_y_mm - act_y_mm) ** 2)
                
                if distance > tolerance:
                    misaligned_components.append({
                        "id": d_id,
                        "type": d.get("type", "Unknown"),
                        "expected_x_pct": exp_x_pct,
                        "expected_y_pct": exp_y_pct,
                        "actual_x_pct": act_x_pct,
                        "actual_y_pct": act_y_pct,
                        "distance_mm": round(distance, 2),
                        "tolerance_mm": tolerance
                    })
                    
        return misaligned_components

