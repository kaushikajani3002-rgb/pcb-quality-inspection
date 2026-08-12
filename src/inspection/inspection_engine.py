from typing import Dict, Any, List
from src.inspection.component_counter import ComponentCounter
from src.inspection.missing_checker import MissingChecker
from src.inspection.extra_checker import ExtraChecker
from src.inspection.position_checker import PositionChecker
from src.inspection.crack_checker import CrackChecker
from src.utils.constants import STATUS_PASS, STATUS_FAIL
from src.utils.logger import logger

class InspectionEngine:
    """
    Main orchestration class that triggers individual checks and 
    aggregates results into a single standardized payload.
    """
    def __init__(self):
        self.counter = ComponentCounter()
        self.missing_checker = MissingChecker()
        self.extra_checker = ExtraChecker()
        self.position_checker = PositionChecker()
        self.crack_checker = CrackChecker()

    def inspect_board(
        self, 
        template: Dict[str, Any], 
        detected_components: List[Dict[str, Any]], 
        position_tolerance: float,
        precalculated_cracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Runs complete visual checks and returns aggregated inspection statistics.
        
        :param template: The reference JSON template.
        :param detected_components: List of component objects detected by YOLO.
        :param position_tolerance: Threshold limit for Euclidean distance.
        :param precalculated_cracks: Solder crack segments list.
        :return: Standardized inspection summary.
        """
        logger.info("InspectionEngine: Starting assembly validation workflow.")
        
        expected_comps = template.get("components", [])

        # 1. Run Checkers
        board_dims = template.get("board_dimensions", {})
        count_stats = self.counter.check(expected_comps, detected_components)
        missing_list = self.missing_checker.check(expected_comps, detected_components)
        extra_list = self.extra_checker.check(expected_comps, detected_components)
        misaligned_list = self.position_checker.check(expected_comps, detected_components, position_tolerance, board_dims)
        crack_list = self.crack_checker.check(detected_components, precalculated_cracks)

        # 2. Determine PASS / FAIL Status
        # If there are any missing, extra, misaligned components, or cracks -> FAIL
        has_anomalies = (
            len(missing_list) > 0 or 
            len(extra_list) > 0 or 
            len(misaligned_list) > 0 or 
            len(crack_list) > 0
        )
        status = STATUS_FAIL if has_anomalies else STATUS_PASS

        logger.info(f"InspectionEngine: Completed. Resulting Status: {status}")

        return {
            "status": status,
            "template_name": template.get("template_name", "Unknown PCB"),
            "component_statistics": {
                "total_expected": count_stats["expected_total"],
                "total_detected": count_stats["detected_total"],
                "by_type": count_stats["expected_by_type"]
            },
            "detected_components": detected_components,
            "missing": missing_list,
            "extra": extra_list,
            "misaligned": misaligned_list,
            "cracks": crack_list
        }
Definition_File_Init: None
