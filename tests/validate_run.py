import sys
from pathlib import Path

# Insert project root into sys.path
project_root = Path(__file__).resolve().parent.parent
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
        print(f"  Loaded '{model_name}' model: {model}")
        if model is None:
            print(f"  ⚠ Model '{model_name}' weights not found on disk (expected if weights are not downloaded).")
    print("✔ Model path resolution succeeded! (weights absent = simulation fallback)")
except Exception as e:
    print(f"❌ Model path check failed: {e}")
    sys.exit(1)

print("\nStep 6: Testing PCB template → defect model mapping...")
try:
    expected_mapping = {
        "arduino_uno": "DeepPCB",
        "esp32_devkit": "DsPCBSD+",
        "stm32_blue_pill": "HRIPCB",
        "generic_pcb": "TDD-PCB",
    }
    all_passed = True
    for template_key, expected_model in expected_mapping.items():
        resolved_model = config.get(f"models.defect_mapping.{template_key}")
        if isinstance(resolved_model, dict):
            resolved_name = resolved_model.get("name")
        else:
            resolved_name = resolved_model
            
        status = "✔" if resolved_name == expected_model else "❌"
        if resolved_name != expected_model:
            all_passed = False
        print(f"  {status} {template_key} → {resolved_name} (expected: {expected_model})")
    if not all_passed:
        raise ValueError("One or more template-to-model mappings did not match.")
    print("✔ All PCB template → defect model mappings verified!")
except Exception as e:
    print(f"❌ Defect mapping test failed: {e}")
    sys.exit(1)

print("\nStep 7: Executing ModelManager Integration Tests (10 Scenarios)...")
try:
    import os
    from unittest.mock import patch, MagicMock
    from src.ai.model_manager import ModelManager
    
    # We patch Path.exists to return True so that validation checks pass for the duration of this test
    # We also patch YOLO to return a dummy string or mock object to verify caching
    with patch('pathlib.Path.exists', return_value=True), \
         patch('src.ai.model_manager.YOLO') as mock_yolo:
        
        mock_yolo.side_effect = lambda path: f"YOLO_Instance_{os.path.basename(path)}"
        
        manager = ModelManager()
        # Reset any cached state
        manager._get_cache().clear()
        
        # Test 10: Configuration validation
        comp_cfg = manager.config.get("models.component_model")
        assert isinstance(comp_cfg, dict) and "path" in comp_cfg, "component_model configuration missing"
        for tkey in ["arduino_uno", "esp32_devkit", "stm32_blue_pill", "generic_pcb"]:
            assert isinstance(manager.config.get(f"models.defect_mapping.{tkey}"), dict), f"mapping for {tkey} missing"
        print("  ✔ Test 10: Configuration validation passed.")

        # Test 1: Arduino resolution
        ard_def = manager.config.get("models.defect_mapping.arduino_uno")
        assert ard_def.get("name") == "DeepPCB", "Arduino defect model name mismatch"
        print("  ✔ Test 1: Arduino configuration lookup passed.")
        
        # Test 2: ESP32 resolution
        esp_def = manager.config.get("models.defect_mapping.esp32_devkit")
        assert esp_def.get("name") == "DsPCBSD+", "ESP32 defect model name mismatch"
        print("  ✔ Test 2: ESP32 configuration lookup passed.")
        
        # Test 3: STM32 resolution
        stm_def = manager.config.get("models.defect_mapping.stm32_blue_pill")
        assert stm_def.get("name") == "HRIPCB", "STM32 defect model name mismatch"
        print("  ✔ Test 3: STM32 configuration lookup passed.")
        
        # Test 4: Generic resolution
        gen_def = manager.config.get("models.defect_mapping.generic_pcb")
        assert gen_def.get("name") == "TDD-PCB", "Generic defect model name mismatch"
        print("  ✔ Test 4: Generic configuration lookup passed.")
        
        # Test 5: Component model consistency
        comp_model_1 = manager.get_component_model()
        comp_model_2 = manager.get_component_model()
        assert comp_model_1 == comp_model_2, "Component model is inconsistent"
        print("  ✔ Test 5: Component model consistency verified.")
        
        # Test 6: Profile switching
        model_uno = manager.get_defect_model("arduino_uno")
        model_esp = manager.get_defect_model("esp32_devkit")
        assert "deeppcb" in model_uno.lower(), "Incorrect model for Arduino"
        assert "dspcbsd" in model_esp.lower(), "Incorrect model for ESP32"
        print("  ✔ Test 6: Profile model switching verified.")
        
        # Test 8: Caching verification
        call_count_before = mock_yolo.call_count
        model_uno_again = manager.get_defect_model("arduino_uno")
        assert model_uno == model_uno_again, "Cache did not reuse defect model object"
        assert mock_yolo.call_count == call_count_before, "Cache called YOLO initializer again"
        print("  ✔ Test 8: Model manager caching verified.")
        
        # Test 9: Model separation
        assert comp_model_1 != model_uno, "Component and defect models are not separated"
        print("  ✔ Test 9: Model separation verified.")
        
    # Test 7: Missing model validation (using actual unpatched Path.exists)
    manager_real = ModelManager()
    try:
        manager_real._load_and_cache("FakeModel", "models/trained/nonexistent_weights.pt", "ContextTest")
        raise AssertionError("Nonexistent weights did not trigger FileNotFoundError")
    except FileNotFoundError as e:
        assert "FakeModel" in str(e), "Error message did not identify the missing model"
        assert "ContextTest" in str(e), "Error message did not identify the context/profile"
        print("  ✔ Test 7: Missing model validation verified (raised FileNotFoundError with context).")
        
    print("✔ All 10 ModelManager integration scenarios verified successfully!")
except Exception as e:
    print(f"❌ ModelManager integration test failed: {e}")
    sys.exit(1)

print("\n🎉 ALL VALIDATIONS PASSED SUCCESSFULLY!")
