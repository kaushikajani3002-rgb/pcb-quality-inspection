import os
import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """
    Utility class to load and validate system configurations from config.yaml.
    """
    def __init__(self, config_path: str = "configs/app.yaml"):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_path = self.project_root / config_path
        self.config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Loads YAML configurations from configs/ directory and merges them.
        """
        configs_dir = self.project_root / "configs"
        config_files = ["app.yaml", "inference.yaml", "model.yaml", "training.yaml"]
        
        self.config_data = {}
        loaded_any = False

        if configs_dir.exists():
            for file_name in config_files:
                file_path = configs_dir / file_name
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            data = yaml.safe_load(file) or {}
                            for k, v in data.items():
                                if k in self.config_data and isinstance(self.config_data[k], dict) and isinstance(v, dict):
                                    self.config_data[k].update(v)
                                else:
                                    self.config_data[k] = v
                            loaded_any = True
                    except Exception:
                        pass
        
        if not loaded_any:
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as file:
                        self.config_data = yaml.safe_load(file) or {}
                        loaded_any = True
                except Exception:
                    pass

        if not loaded_any:
            self.config_data = {
                "inspection": {
                    "confidence": 0.50,
                    "iou": 0.45,
                    "position_tolerance": 15.0
                },
                "paths": {
                    "template_folder": "templates",
                    "report_folder": "outputs/reports",
                    "log_folder": "logs",
                    "output_folder": "outputs/predictions"
                },
                "dashboard": {
                    "theme": "dark",
                    "default_operator": "Operator_AOI_04",
                    "company_name": "Anti-Gravity PCB Assembly Line",
                    "logo_path": "assets/logo_placeholder.png"
                }
            }
            
        return self.config_data

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Fetches values using dot notation (e.g., 'inspection.confidence').
        """
        keys = key_path.split(".")
        current = self.config_data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_resolved_path(self, path_key: str) -> Path:
        """
        Resolves folder paths from the configuration relative to the project root.
        """
        folder_name = self.get(f"paths.{path_key}")
        if not folder_name:
            raise KeyError(f"Path key '{path_key}' not defined in config paths.")
        
        resolved = self.project_root / folder_name
        return resolved
