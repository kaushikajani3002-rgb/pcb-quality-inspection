import os
import sys
from pathlib import Path
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.detection_engine import load_model, run_component_counting
from src.utils.template_manager import TemplateManager

def test_dual_inference():
    print("Loading component and defect models...")
    comp_model = load_model("Component")
    defect_model = load_model("DeepPCB")
    
    if not comp_model or not defect_model:
        print("❌ Error: Models could not be loaded.")
        sys.exit(1)
        
    print("Models loaded successfully.")
    
    # Create a blank 640x480 image
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    
    # Load template
    tm = TemplateManager()
    template = tm.load_template("arduino_uno")
    
    print("Running dual inference...")
    results = run_component_counting(
        uploaded_image=img,
        component_model=comp_model,
        defect_model=defect_model,
        conf_slider=0.25,
        iou_slider=0.45,
        active_template=template,
        position_tolerance_slider=1.5,
        defect_mode=False
    )
    
    print("\nInference Results:")
    print(f"  - Status: {results['status']}")
    print(f"  - Processing Time: {results['processing_time']}")
    print(f"  - Detected Components: {len(results['detected_components'])}")
    print(f"  - Missing Components: {len(results['missing'])}")
    print(f"  - Extra Components: {len(results['extra'])}")
    print(f"  - Misaligned Components: {len(results['misaligned'])}")
    print(f"  - Detected PCB Defects: {len(results['cracks'])}")
    
    if len(results['detected_components']) > 0:
        print("\nComponent Detections Detail:")
        for det in results['detected_components'][:5]:
            print(f"    - Type: {det['type']}, Center: ({det['center_x_pct']:.2f}, {det['center_y_pct']:.2f}), Confidence: {det.get('confidence', 0.0)}")
            
    print("\n✔ Inference completed successfully!")

if __name__ == "__main__":
    test_dual_inference()
