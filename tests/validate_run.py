import sys
from pathlib import Path

# Insert project root into sys.path
project_root = Path(r"d:\Pcb Quality Inspection")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Step 1: Testing imports...")
try:
    from src.utils.config_loader import ConfigLoader
    from src.utils.template_manager import TemplateManager
    from src.utils.logger import logger
    from src.utils.report_exporter import ReportExporterFactory
    from src.utils.validators import ImageValidator, ConfigurationValidator
    from src.mock.mock_results import MockInspectionService
    from src.inspection.inspection_engine import InspectionEngine
    from src.ai.detection_engine import load_model, run_component_counting
    print("✔ All imports succeeded!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("\nStep 2: Testing ConfigLoader and path resolving...")
try:
    config = ConfigLoader()
    print("Resolved configuration keys:")
    print(f"  - Theme: {config.get('dashboard.theme')}")
    print(f"  - Default Operator: {config.get('dashboard.default_operator')}")
    print(f"  - Confidence: {config.get('inspection.confidence')}")
    print(f"  - Report Folder: {config.get_resolved_path('report_folder')}")
    print(f"  - Prediction Folder: {config.get_resolved_path('output_folder')}")
    print("✔ Configuration loading and merging succeeded!")
except Exception as e:
    print(f"❌ ConfigLoader failed: {e}")
    sys.exit(1)

print("\nStep 3: Testing TemplateManager...")
try:
    tm = TemplateManager()
    templates = tm.list_templates()
    print(f"Available templates: {templates}")
    if not templates:
        raise ValueError("No templates found!")
    
    # Load Arduino Uno template
    uno_template = tm.load_template("arduino_uno")
    print(f"Loaded 'arduino_uno' template containing {uno_template['total_critical_components']} components.")
    print("✔ TemplateManager loading and listing succeeded!")
except Exception as e:
    print(f"❌ TemplateManager failed: {e}")
    sys.exit(1)

print("\nStep 4: Testing Mock Inspection pipeline...")
try:
    mock_results = MockInspectionService.run_inspection(
        template=uno_template,
        defect_mode=True,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        position_tolerance=15.0
    )
    print(f"Mock inspection completed: Status={mock_results.get('status')}, Issues={mock_results.get('total_issues')}")
    print("✔ Mock Inspection pipeline succeeded!")
except Exception as e:
    print(f"❌ Mock Inspection failed: {e}")
    sys.exit(1)

print("\nStep 5: Testing AI detection model path checking for all models...")
try:
    for model_name in ["Component", "DeepPCB", "DsPCBSD+", "HRIPCB", "TDD-PCB"]:
        model = load_model(model_name)
        print(f"Loaded '{model_name}' model successfully: {model}")
        if model is None:
            raise ValueError(f"Failed to load model {model_name}")
    print("✔ Model path validation for all models succeeded!")
except Exception as e:
    print(f"❌ Model path check failed: {e}")
    sys.exit(1)

print("\n🎉 ALL VALIDATIONS PASSED SUCCESSFULLY!")
