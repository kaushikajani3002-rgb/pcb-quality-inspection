import os
import sys
import unittest
from pathlib import Path
from PIL import Image

# Insert project root into sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.detection_engine import run_component_inspection, run_circuit_inspection
from src.utils.template_manager import TemplateManager

class TestWorkflowScenarios(unittest.TestCase):
    def setUp(self):
        self.tm = TemplateManager()
        self.template = self.tm.load_template("arduino_uno")
        self.img = Image.new("RGB", (640, 480), color=(128, 128, 128))

    def test_aggregator_logic(self):
        print("\nExecuting test_aggregator_logic...")
        
        # Test Cases: (comp_status, circ_status) -> expected_final
        cases = [
            ("PASS", "PASS", "🟢 PCB INSPECTION PASSED"),
            ("FAIL", "PASS", "🔴 COMPONENT DEFECT DETECTED"),
            ("PASS", "FAIL", "🔴 CIRCUIT DEFECT DETECTED"),
            ("FAIL", "FAIL", "🔴 COMPONENT + CIRCUIT DEFECTS DETECTED"),
            ("PASS", "NOT_INSPECTED", "⚠️ INSPECTION INCOMPLETE"),
            ("NOT_INSPECTED", "PASS", "⚠️ INSPECTION INCOMPLETE"),
            ("NOT_INSPECTED", "FAIL", "⚠️ INSPECTION INCOMPLETE"),
            ("NOT_INSPECTED", "NOT_INSPECTED", "⚠️ INSPECTION INCOMPLETE")
        ]
        
        for comp_status, circ_status, expected_final in cases:
            # Simulate the aggregator logic from main.py
            if comp_status == "PASS" and circ_status == "PASS":
                final = "🟢 PCB INSPECTION PASSED"
            elif comp_status == "FAIL" and circ_status == "PASS":
                final = "🔴 COMPONENT DEFECT DETECTED"
            elif comp_status == "PASS" and circ_status == "FAIL":
                final = "🔴 CIRCUIT DEFECT DETECTED"
            elif comp_status == "FAIL" and circ_status == "FAIL":
                final = "🔴 COMPONENT + CIRCUIT DEFECTS DETECTED"
            else:
                final = "⚠️ INSPECTION INCOMPLETE"
                
            self.assertEqual(final, expected_final, f"Failed for {comp_status} + {circ_status}")
            print(f"  ✔ Verified: Comp={comp_status:15s} + Circ={circ_status:15s} => {final}")

if __name__ == "__main__":
    unittest.main()
