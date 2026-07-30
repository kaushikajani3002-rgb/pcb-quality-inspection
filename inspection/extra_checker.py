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
            # If the ID is not in template, or component is explicitly marked extra
            if d_id not in expected_ids or d.get("status") == "Extra":
                extra_components.append({
                    "id": d_id,
                    "type": d.get("type", "Unknown"),
                    "center_x": d.get("center_x"),
                    "center_y": d.get("center_y"),
                    "width": d.get("width"),
                    "height": d.get("height"),
                    "confidence": d.get("confidence", 0.0),
                    "reason": "Unregistered component ID or unexpected placement location"
                })
                
        return extra_components
