from typing import Dict, Any, List

class MissingChecker:
    """
    Checks if components defined in the template are missing from the detections.
    """
    def __init__(self):
        pass

    def check(self, expected: List[Dict[str, Any]], detected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans expected PCB template coordinates and verifies whether a matching 
        element was found within close range in the detections.
        
        :param expected: List of expected component dicts.
        :param detected: List of detected component dicts.
        :return: List of missing components details.
        """
        missing_components: List[Dict[str, Any]] = []
        
        # Spatial threshold matching (mock algorithm placeholder for CV integration)
        detected_ids = {d.get("id") for d in detected if d.get("id")}
        
        for e in expected:
            e_id = e.get("id")
            if e_id not in detected_ids:
                missing_components.append({
                    "id": e_id,
                    "type": e.get("type", "Unknown"),
                    "expected_x_pct": e.get("center_x_pct", 0.0),
                    "expected_y_pct": e.get("center_y_pct", 0.0),
                    "width_pct": e.get("width_pct", 0.0),
                    "height_pct": e.get("height_pct", 0.0),
                    "expected_x": e.get("center_x", 0.0),
                    "expected_y": e.get("center_y", 0.0),
                    "width": e.get("width", 0.0),
                    "height": e.get("height", 0.0),
                    "reason": "Missing component at expected coordinates"
                })
                
        return missing_components
