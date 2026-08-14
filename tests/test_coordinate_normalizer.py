import unittest
from src.ai.detection_engine import normalize_component_coordinates

class TestCoordinateNormalizer(unittest.TestCase):
    def test_normalize_expected_component_ratio(self):
        # Simulates a component loaded from reference JSON template (using 0-1 ratios)
        comp = {
            "id": "C1",
            "type": "Capacitor",
            "center_x_pct": 0.55,
            "center_y_pct": 0.45,
            "width_pct": 0.10,
            "height_pct": 0.20
        }
        res = normalize_component_coordinates(comp, 600, 400)
        
        self.assertEqual(res["id"], "C1")
        self.assertEqual(res["type"], "Capacitor")
        # Pixel coordinates
        self.assertEqual(res["center_x"], 330.0) # 0.55 * 600
        self.assertEqual(res["center_y"], 180.0) # 0.45 * 400
        self.assertEqual(res["width"], 60.0)      # 0.10 * 600
        self.assertEqual(res["height"], 80.0)     # 0.20 * 400
        
        # Percentage coordinates (0-100 scale)
        self.assertEqual(res["center_x_pct"], 55.0)
        self.assertEqual(res["center_y_pct"], 45.0)
        self.assertEqual(res["width_pct"], 10.0)
        self.assertEqual(res["height_pct"], 20.0)
        
        # Bounding box bounds
        self.assertEqual(res["x1"], 300.0) # 330 - 30
        self.assertEqual(res["y1"], 140.0) # 180 - 40
        self.assertEqual(res["x2"], 360.0) # 330 + 30
        self.assertEqual(res["y2"], 220.0) # 180 + 40

    def test_normalize_expected_component_percentage(self):
        # Simulates a component loaded from reference JSON template (already in 0-100 scale)
        comp = {
            "id": "R1",
            "type": "Resistor",
            "center_x_pct": 75.0,
            "center_y_pct": 25.0,
            "width_pct": 8.0,
            "height_pct": 12.0
        }
        res = normalize_component_coordinates(comp, 1000, 1000)
        
        self.assertEqual(res["center_x"], 750.0)
        self.assertEqual(res["center_y"], 250.0)
        self.assertEqual(res["center_x_pct"], 75.0)
        self.assertEqual(res["center_y_pct"], 25.0)

    def test_normalize_yolo_detection_pixels(self):
        # Simulates a component detected by YOLO (pixel bounds)
        comp = {
            "type": "IC",
            "x1": 100,
            "y1": 200,
            "x2": 300,
            "y2": 400,
            "confidence": 0.85
        }
        res = normalize_component_coordinates(comp, 1000, 800)
        
        self.assertEqual(res["center_x"], 200.0)
        self.assertEqual(res["center_y"], 300.0)
        self.assertEqual(res["width"], 200.0)
        self.assertEqual(res["height"], 200.0)
        
        # Percentage coordinates (0-100 scale)
        self.assertEqual(res["center_x_pct"], 20.0) # (200/1000)*100
        self.assertEqual(res["center_y_pct"], 37.5) # (300/800)*100
        
if __name__ == "__main__":
    unittest.main()
