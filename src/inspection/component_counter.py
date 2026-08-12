from typing import Dict, Any, List

class ComponentCounter:
    """
    Checks component frequencies and balances expected vs detected counts.
    """
    def __init__(self):
        pass

    def check(self, expected: List[Dict[str, Any]], detected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compiles count statistics comparing the expected template layout 
        against the detected object instances.
        
        :param expected: List of expected component dicts.
        :param detected: List of detected component dicts.
        :return: A dictionary containing totals and component type frequency comparisons.
        """
        expected_counts: Dict[str, int] = {}
        for comp in expected:
            comp_type = comp.get("type", "Unknown")
            expected_counts[comp_type] = expected_counts.get(comp_type, 0) + 1

        detected_counts: Dict[str, int] = {}
        for comp in detected:
            # Only count registered non-extra components
            if comp.get("status") != "Extra":
                comp_type = comp.get("type", "Unknown")
                detected_counts[comp_type] = detected_counts.get(comp_type, 0) + 1

        # Calculate difference matrix
        difference: Dict[str, int] = {}
        all_types = set(expected_counts.keys()).union(detected_counts.keys())
        for c_type in all_types:
            exp_val = expected_counts.get(c_type, 0)
            det_val = detected_counts.get(c_type, 0)
            difference[c_type] = det_val - exp_val

        return {
            "expected_total": len(expected),
            "detected_total": len(detected),
            "expected_by_type": expected_counts,
            "detected_by_type": detected_counts,
            "count_difference": difference
        }
