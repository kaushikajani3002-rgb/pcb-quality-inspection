import os
from pathlib import Path
from typing import Any, Dict, List
from src.utils.json_loader import JsonLoader, TEMPLATES_DIR, FALLBACK_TEMPLATE
from src.utils.logger import logger

class TemplateManager:
    """
    Manages PCB templates including listing, loading, validating, 
    saving, and creating expected component layouts. Uses globally resolved absolute pathing.
    """
    def __init__(self, templates_dir: str = None):
        self.templates_dir = Path(TEMPLATES_DIR)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-initialize and write all three profiles to disk on startup if they don't exist
        for stem in ["arduino_uno", "esp32_devkit", "stm32_blue_pill", "generic_pcb"]:
            self.load_template(stem)

    def list_templates(self) -> List[str]:
        """
        Lists all templates (.json files) available in the templates folder.
        """
        try:
            files = list(self.templates_dir.glob("*.json"))
            stems = [f.stem for f in files]
            
            # Ensure our three primary options are always in the list
            for primary in ["arduino_uno", "esp32_devkit", "stm32_blue_pill", "generic_pcb"]:
                if primary not in stems:
                    stems.append(primary)
            return sorted(list(set(stems)))
        except Exception as e:
            logger.error(f"Error listing templates in {self.templates_dir}: {e}")
            return ["arduino_uno", "esp32_devkit", "generic_pcb", "stm32_blue_pill"]

    def load_template(self, template_name: str) -> Dict[str, Any]:
        """
        Loads a specific template by name. Returns a valid default layout if not found.
        """
        file_path = self.templates_dir / f"{template_name}.json"
        
        # Load via the exception-safe loader (which creates default layouts on-the-fly)
        data = JsonLoader.load(file_path)
        
        # Validate schema. If validation fails, return fallback template
        if not self.validate_template(data):
            logger.warning(f"Validation failed for template '{template_name}'. Using fallback.")
            return FALLBACK_TEMPLATE
            
        return data

    def validate_template(self, template_data: Dict[str, Any]) -> bool:
        """
        Verifies schema compliance of a loaded template dictionary.
        Checks for fields like board_name, board_dimensions, total_critical_components, and components.
        """
        if not isinstance(template_data, dict):
            return False

        required_root_keys = {"board_name", "board_dimensions", "total_critical_components", "components"}
        required_dim_keys = {"width_mm", "height_mm", "pixel_width", "pixel_height"}
        required_comp_keys = {"id", "type", "center_x_pct", "center_y_pct", "width_pct", "height_pct"}

        # Validate Root Keys
        if not required_root_keys.issubset(template_data.keys()):
            missing = required_root_keys - template_data.keys()
            logger.warning(f"Template validation failed. Missing root keys: {missing}")
            return False

        # Validate Board Dimensions
        dimensions = template_data.get("board_dimensions", {})
        if not isinstance(dimensions, dict) or not required_dim_keys.issubset(dimensions.keys()):
            missing = required_dim_keys - dimensions.keys() if isinstance(dimensions, dict) else "Not a dict"
            logger.warning(f"Template validation failed. Missing board_dimensions keys: {missing}")
            return False

        # Validate Component Definitions
        components = template_data.get("components", [])
        if not isinstance(components, list):
            logger.warning("Template validation failed. 'components' must be a list.")
            return False

        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                logger.warning(f"Component at index {idx} is not a valid dictionary.")
                return False
            if not required_comp_keys.issubset(comp.keys()):
                missing = required_comp_keys - comp.keys()
                logger.warning(f"Component at index {idx} ({comp.get('id', 'unknown')}) is missing keys: {missing}")
                return False

        return True

    def save_template(self, template_name: str, template_data: Dict[str, Any]) -> bool:
        """
        Saves a template dictionary to a JSON file.
        """
        if not self.validate_template(template_data):
            logger.error("Failed to save template. Template structure is invalid.")
            return False
        
        file_path = self.templates_dir / f"{template_name}.json"
        return JsonLoader.save(template_data, file_path)

    def create_template(
        self, 
        name: str, 
        width_mm: float, 
        height_mm: float, 
        pixel_width: int,
        pixel_height: int,
        components: List[Dict[str, Any]], 
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Creates a template dictionary dynamically.
        """
        new_template = {
            "board_name": name,
            "board_dimensions": {
                "width_mm": width_mm,
                "height_mm": height_mm,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height
            },
            "total_critical_components": len(components),
            "description": description,
            "components": components
        }
        return new_template
