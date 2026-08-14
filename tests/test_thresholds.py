import os
import sys
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock

# Insert project root into sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.detection_engine import run_component_counting
from src.utils.template_manager import TemplateManager

def test_thresholds_passing():
    print("Initializing test_thresholds_passing...")
    
    # Mock YOLO component model
    mock_component_model = MagicMock()
    mock_predict_result = MagicMock()
    mock_predict_result.boxes = []
    mock_component_model.predict.return_value = [mock_predict_result]
    
    # Mock YOLO defect model
    mock_defect_model = MagicMock()
    mock_defect_model.predict.return_value = [mock_predict_result]
    
    # Load template
    tm = TemplateManager()
    template = tm.load_template("arduino_uno")
    
    # Create target dummy image
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    
    # Test cases for Confidence
    conf_values = [0.20, 0.50, 0.80]
    for conf in conf_values:
        # Reset mock calls
        mock_component_model.predict.reset_mock()
        mock_defect_model.predict.reset_mock()
        
        run_component_counting(
            uploaded_image=img,
            component_model=mock_component_model,
            defect_model=mock_defect_model,
            conf_slider=conf,
            iou_slider=0.45,
            active_template=template,
            position_tolerance_slider=1.5,
            defect_mode=True
        )
        
        # Verify component model received correct conf parameter
        comp_args, comp_kwargs = mock_component_model.predict.call_args
        assert comp_kwargs.get("conf") == conf, f"Component model did not receive conf={conf}, got {comp_kwargs.get('conf')}"
        assert comp_kwargs.get("iou") == 0.45, f"Component model did not receive iou=0.45, got {comp_kwargs.get('iou')}"
        
        # Verify defect model received correct conf parameter
        def_args, def_kwargs = mock_defect_model.predict.call_args
        assert def_kwargs.get("conf") == conf, f"Defect model did not receive conf={conf}, got {def_kwargs.get('conf')}"
        assert def_kwargs.get("iou") == 0.45, f"Defect model did not receive iou=0.45, got {def_kwargs.get('iou')}"
        
        print(f"  ✔ Confidence threshold test passed for conf={conf}")
        
    # Test cases for IoU
    iou_values = [0.20, 0.80]
    for iou in iou_values:
        # Reset mock calls
        mock_component_model.predict.reset_mock()
        mock_defect_model.predict.reset_mock()
        
        run_component_counting(
            uploaded_image=img,
            component_model=mock_component_model,
            defect_model=mock_defect_model,
            conf_slider=0.50,
            iou_slider=iou,
            active_template=template,
            position_tolerance_slider=1.5,
            defect_mode=True
        )
        
        # Verify component model received correct iou parameter
        comp_args, comp_kwargs = mock_component_model.predict.call_args
        assert comp_kwargs.get("conf") == 0.50, f"Component model did not receive conf=0.50, got {comp_kwargs.get('conf')}"
        assert comp_kwargs.get("iou") == iou, f"Component model did not receive iou={iou}, got {comp_kwargs.get('iou')}"
        
        # Verify defect model received correct iou parameter
        def_args, def_kwargs = mock_defect_model.predict.call_args
        assert def_kwargs.get("conf") == 0.50, f"Defect model did not receive conf=0.50, got {def_kwargs.get('conf')}"
        assert def_kwargs.get("iou") == iou, f"Defect model did not receive iou={iou}, got {def_kwargs.get('iou')}"
        
        print(f"  ✔ IoU threshold test passed for iou={iou}")
        
    print("\n🎉 ALL INFERENCE THRESHOLDS PROPAGATE CORRECTLY!")

if __name__ == "__main__":
    test_thresholds_passing()
