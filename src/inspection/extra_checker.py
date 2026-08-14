from typing import Dict, Any, List

class ExtraChecker:
    """
    Checks if there are any detected components that are not defined in the template.
    """
    def __init__(self):
        pass

    def check(self, expected: List[Dict[str, Any]], detected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cross-references detected components with template listings. Any component
        found on the board that does not correlate to an expected component is flagged.
        
        :param expected: List of expected template components.
        :param detected: List of detected components.
        :return: List of extra components found.
        """
        extra_components: List[Dict[str, Any]] = []
        expected_ids = {e.get("id") for e in expected if e.get("id")}
        
        for d in detected:
            d_id = d.get("id")
            # Ignore the PCB board itself
            if str(d.get("type", "")).lower() == "pcb" or str(d.get("class_name", "")).lower() == "pcb":
                continue
            # If the ID is not in template, or component is explicitly marked extra
            if d_id not in expected_ids or d.get("status") == "Extra":
                extra_components.append({
                    "id": d_id,
                    "type": d.get("type", "Unknown"),
                    "center_x_pct": d.get("center_x_pct", 0.0),
                    "center_y_pct": d.get("center_y_pct", 0.0),
                    "width_pct": d.get("width_pct", 0.0),
                    "height_pct": d.get("height_pct", 0.0),
                    "center_x": d.get("center_x", 0.0),
                    "center_y": d.get("center_y", 0.0),
                    "width": d.get("width", 0.0),
                    "height": d.get("height", 0.0),
                    "confidence": d.get("confidence", 0.0),
                    "reason": "Unregistered component ID or unexpected placement location"
                })
                
        return extra_components
