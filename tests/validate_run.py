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
    from src.ai.model_manager import ModelManager
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

print("\nStep 5: Testing backward-compatible load_model() path resolution...")
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

print("\nStep 6: Testing PCB template → defect model mapping (new dict config)...")
try:
    expected_mapping = {
        "arduino_uno": "DeepPCB",
        "esp32_devkit": "DsPCBSD+",
        "stm32_blue_pill": "HRIPCB",
        "generic_pcb": "TDD-PCB",
    }
    all_passed = True
    for template_key, expected_model in expected_mapping.items():
        mapping = config.get(f"models.defect_mapping.{template_key}")
        if isinstance(mapping, dict):
            resolved_model = mapping.get("name")
        else:
            resolved_model = mapping
        status = "✔" if resolved_model == expected_model else "❌"
        if resolved_model != expected_model:
            all_passed = False
        print(f"  {status} {template_key} → {resolved_model} (expected: {expected_model})")
    if not all_passed:
        raise ValueError("One or more template-to-model mappings did not match.")
    print("✔ All PCB template → defect model mappings verified!")
except Exception as e:
    print(f"❌ Defect mapping test failed: {e}")
    sys.exit(1)

# -------------------------------------------------------------------------
# MODEL MANAGER TESTS (Steps 7-14)
# -------------------------------------------------------------------------

print("\nStep 7: Testing ModelManager — Arduino Uno → DeepPCB + Component...")
try:
    mgr = ModelManager()
    defect_model, defect_name = mgr.get_defect_model("arduino_uno")
    comp_model = mgr.get_component_model()
    defect_info = mgr.get_defect_model_info("arduino_uno")
    comp_info = mgr.get_component_model_info()
    assert defect_info["name"] == "DeepPCB", f"Expected DeepPCB, got {defect_info['name']}"
    assert comp_info["name"] == "Component", f"Expected Component, got {comp_info['name']}"
    print(f"  Defect: {defect_info['name']} → {defect_info['path']}")
    print(f"  Component: {comp_info['name']} → {comp_info['path']}")
    print("✔ Arduino Uno model resolution PASSED!")
except Exception as e:
    print(f"❌ Arduino Uno test failed: {e}")
    sys.exit(1)

print("\nStep 8: Testing ModelManager — ESP32 DevKit → DsPCBSD+ + Component...")
try:
    defect_info = mgr.get_defect_model_info("esp32_devkit")
    assert defect_info["name"] == "DsPCBSD+", f"Expected DsPCBSD+, got {defect_info['name']}"
    assert comp_info["name"] == "Component"
    print(f"  Defect: {defect_info['name']} → {defect_info['path']}")
    print(f"  Component: {comp_info['name']} (unchanged)")
    print("✔ ESP32 DevKit model resolution PASSED!")
except Exception as e:
    print(f"❌ ESP32 DevKit test failed: {e}")
    sys.exit(1)

print("\nStep 9: Testing ModelManager — STM32 Blue Pill → HRIPCB + Component...")
try:
    defect_info = mgr.get_defect_model_info("stm32_blue_pill")
    assert defect_info["name"] == "HRIPCB", f"Expected HRIPCB, got {defect_info['name']}"
    assert comp_info["name"] == "Component"
    print(f"  Defect: {defect_info['name']} → {defect_info['path']}")
    print(f"  Component: {comp_info['name']} (unchanged)")
    print("✔ STM32 Blue Pill model resolution PASSED!")
except Exception as e:
    print(f"❌ STM32 Blue Pill test failed: {e}")
    sys.exit(1)

print("\nStep 10: Testing ModelManager — Generic PCB → TDD-PCB + Component...")
try:
    defect_info = mgr.get_defect_model_info("generic_pcb")
    assert defect_info["name"] == "TDD-PCB", f"Expected TDD-PCB, got {defect_info['name']}"
    assert comp_info["name"] == "Component"
    print(f"  Defect: {defect_info['name']} → {defect_info['path']}")
    print(f"  Component: {comp_info['name']} (unchanged)")
    print("✔ Generic PCB model resolution PASSED!")
except Exception as e:
    print(f"❌ Generic PCB test failed: {e}")
    sys.exit(1)

print("\nStep 11: Testing component model consistency across all profiles...")
try:
    comp_path = mgr.get_component_model_info()["path"]
    for profile in ["arduino_uno", "esp32_devkit", "stm32_blue_pill", "generic_pcb"]:
        current_comp = mgr.get_component_model_info()
        assert current_comp["path"] == comp_path, f"Component path changed for {profile}!"
        assert current_comp["name"] == "Component", f"Component name changed for {profile}!"
    print(f"  Component model is consistent: {comp_path}")
    print("✔ Component model consistency PASSED!")
except Exception as e:
    print(f"❌ Component consistency test failed: {e}")
    sys.exit(1)

print("\nStep 12: Testing profile switching (defect model changes, component stays)...")
try:
    expected_sequence = [
        ("arduino_uno", "DeepPCB"),
        ("esp32_devkit", "DsPCBSD+"),
        ("stm32_blue_pill", "HRIPCB"),
        ("generic_pcb", "TDD-PCB"),
        ("arduino_uno", "DeepPCB"),  # back to first
    ]
    for profile, expected_defect in expected_sequence:
        info = mgr.get_defect_model_info(profile)
        assert info["name"] == expected_defect, f"{profile}: expected {expected_defect}, got {info['name']}"
        print(f"  ✔ {profile} → {info['name']}")
    print("✔ Profile switching PASSED!")
except Exception as e:
    print(f"❌ Profile switching test failed: {e}")
    sys.exit(1)

print("\nStep 13: Testing missing model handling...")
try:
    bogus_model, bogus_name = mgr.get_defect_model("nonexistent_board_xyz")
    assert bogus_model is None, "Expected None for nonexistent profile"
    assert bogus_name is None, "Expected None name for nonexistent profile"
    print("  ✔ Nonexistent profile returns (None, None)")
    print("✔ Missing model handling PASSED!")
except Exception as e:
    print(f"❌ Missing model test failed: {e}")
    sys.exit(1)

print("\nStep 14: Testing model caching (same object returned on repeated calls)...")
try:
    # Component model caching
    c1 = mgr.get_component_model()
    c2 = mgr.get_component_model()
    if c1 is not None:
        assert c1 is c2, "Component model was reloaded instead of reused!"
        print("  ✔ Component model: same cached object returned")
    else:
        print("  ⚠ Component model weights absent — cache test skipped (expected)")

    # Defect model caching
    d1, _ = mgr.get_defect_model("arduino_uno")
    d2, _ = mgr.get_defect_model("arduino_uno")
    if d1 is not None:
        assert d1 is d2, "Defect model was reloaded instead of reused!"
        print("  ✔ Defect model (DeepPCB): same cached object returned")
    else:
        print("  ⚠ Defect model weights absent — cache test skipped (expected)")
    print("✔ Model caching PASSED!")
except Exception as e:
    print(f"❌ Model caching test failed: {e}")
    sys.exit(1)

print("\nStep 15: Testing model separation (defect ≠ component)...")
try:
    comp_info = mgr.get_component_model_info()
    for profile in ["arduino_uno", "esp32_devkit", "stm32_blue_pill", "generic_pcb"]:
        def_info = mgr.get_defect_model_info(profile)
        assert def_info["name"] != comp_info["name"], f"Defect model name matches component for {profile}!"
        assert def_info["path"] != comp_info["path"], f"Defect model path matches component for {profile}!"
        print(f"  ✔ {profile}: {def_info['name']} ≠ {comp_info['name']}")
    print("✔ Model separation PASSED!")
except Exception as e:
    print(f"❌ Model separation test failed: {e}")
    sys.exit(1)

print("\nStep 16: Testing configuration validation (all 5 model configs exist)...")
try:
    comp_cfg = config.get("models.component_model")
    assert comp_cfg is not None, "models.component_model missing"
    assert isinstance(comp_cfg, dict), "models.component_model must be a dict"
    assert "name" in comp_cfg and "path" in comp_cfg, "component_model missing name/path"
    print(f"  ✔ component_model: {comp_cfg['name']} → {comp_cfg['path']}")

    for profile in ["arduino_uno", "esp32_devkit", "stm32_blue_pill", "generic_pcb"]:
        m = config.get(f"models.defect_mapping.{profile}")
        assert m is not None, f"defect_mapping.{profile} missing"
        assert isinstance(m, dict), f"defect_mapping.{profile} must be a dict"
        assert "name" in m and "path" in m, f"defect_mapping.{profile} missing name/path"
        print(f"  ✔ {profile}: {m['name']} → {m['path']}")
    print("✔ Configuration validation PASSED!")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    sys.exit(1)

print("\n🎉 ALL VALIDATIONS PASSED SUCCESSFULLY!")
