import os
import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """
    Utility class to load and validate system configurations from config.yaml.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / config_path
        self.config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Loads the YAML configuration file and returns its dictionary representation.
        Raises FileNotFoundError if the config file does not exist.
        """
        if not self.config_path.exists():
            # If default doesn't exist, try local relative search as fallback
            fallback_path = Path("config/config.yaml")
            if fallback_path.exists():
                self.config_path = fallback_path.resolve()
            else:
                raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                self.config_data = yaml.safe_load(file) or {}
            return self.config_data
        except Exception as e:
            # Fallback configuration structure in case of parsing errors
            self.config_data = {
                "inspection": {
                    "confidence": 0.50,
                    "iou": 0.45,
                    "position_tolerance": 15.0
                },
                "paths": {
                    "template_folder": "templates",
                    "report_folder": "reports",
                    "log_folder": "logs",
                    "output_folder": "outputs"
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
