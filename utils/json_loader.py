import json
import os
from pathlib import Path
from typing import Any, Dict, List
from utils.logger import logger

# 1. Absolute Path Routing calculations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Create templates folder if not existing
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Standard structural fallbacks for dynamic generation on the fly
ARDUINO_UNO_DEFAULT = {
    "board_name": "Arduino Uno",
    "board_dimensions": {
        "width_mm": 68.6,
        "height_mm": 53.4,
        "pixel_width": 640,
        "pixel_height": 480
    },
    "total_critical_components": 14,
    "description": "Standard Arduino Uno Rev3 PCB template mapping 14 critical components.",
    "components": [
        {"id": "main_mcu_atmega328p", "type": "IC", "center_x_pct": 0.55, "center_y_pct": 0.45, "width_pct": 0.10, "height_pct": 0.35, "tolerance_override_mm": 1.5},
        {"id": "usb_interface_mcu", "type": "IC", "center_x_pct": 0.22, "center_y_pct": 0.25, "width_pct": 0.08, "height_pct": 0.12},
        {"id": "usb_type_b_port", "type": "Connector", "center_x_pct": 0.12, "center_y_pct": 0.25, "width_pct": 0.20, "height_pct": 0.15},
        {"id": "dc_barrel_jack", "type": "Connector", "center_x_pct": 0.15, "center_y_pct": 0.75, "width_pct": 0.22, "height_pct": 0.20},
        {"id": "voltage_reg_5v", "type": "IC", "center_x_pct": 0.35, "center_y_pct": 0.70, "width_pct": 0.07, "height_pct": 0.12},
        {"id": "voltage_reg_3v3", "type": "IC", "center_x_pct": 0.38, "center_y_pct": 0.80, "width_pct": 0.05, "height_pct": 0.08},
        {"id": "crystal_16mhz", "type": "Connector", "center_x_pct": 0.42, "center_y_pct": 0.28, "width_pct": 0.06, "height_pct": 0.04},
        {"id": "reset_button", "type": "Connector", "center_x_pct": 0.88, "center_y_pct": 0.12, "width_pct": 0.08, "height_pct": 0.10},
        {"id": "female_header_strip_1", "type": "Connector", "center_x_pct": 0.60, "center_y_pct": 0.05, "width_pct": 0.30, "height_pct": 0.05},
        {"id": "female_header_strip_2", "type": "Connector", "center_x_pct": 0.90, "center_y_pct": 0.05, "width_pct": 0.20, "height_pct": 0.05},
        {"id": "female_header_strip_3", "type": "Connector", "center_x_pct": 0.65, "center_y_pct": 0.95, "width_pct": 0.30, "height_pct": 0.05},
        {"id": "female_header_strip_4", "type": "Connector", "center_x_pct": 0.90, "center_y_pct": 0.95, "width_pct": 0.15, "height_pct": 0.05},
        {"id": "power_led", "type": "LED", "center_x_pct": 0.82, "center_y_pct": 0.30, "width_pct": 0.03, "height_pct": 0.04},
        {"id": "pin13_led", "type": "LED", "center_x_pct": 0.82, "center_y_pct": 0.22, "width_pct": 0.03, "height_pct": 0.04}
    ]
}

ESP32_DEVKIT_DEFAULT = {
    "board_name": "ESP32 DevKit",
    "board_dimensions": {
        "width_mm": 48.3,
        "height_mm": 28.9,
        "pixel_width": 640,
        "pixel_height": 480
    },
    "total_critical_components": 9,
    "description": "Standard ESP32 DevKit PCB template mapping 9 critical components.",
    "components": [
        {"id": "esp_wroom_32_module", "type": "IC", "center_x_pct": 0.50, "center_y_pct": 0.28, "width_pct": 0.35, "height_pct": 0.32, "tolerance_override_mm": 1.2},
        {"id": "uart_bridge_ic", "type": "IC", "center_x_pct": 0.50, "center_y_pct": 0.65, "width_pct": 0.12, "height_pct": 0.12},
        {"id": "micro_usb_port", "type": "Connector", "center_x_pct": 0.50, "center_y_pct": 0.88, "width_pct": 0.15, "height_pct": 0.10},
        {"id": "voltage_reg_3v3", "type": "IC", "center_x_pct": 0.28, "center_y_pct": 0.65, "width_pct": 0.08, "height_pct": 0.10},
        {"id": "en_reset_button", "type": "Connector", "center_x_pct": 0.32, "center_y_pct": 0.78, "width_pct": 0.08, "height_pct": 0.08},
        {"id": "boot_button", "type": "Connector", "center_x_pct": 0.68, "center_y_pct": 0.78, "width_pct": 0.08, "height_pct": 0.08},
        {"id": "male_header_strip_left", "type": "Connector", "center_x_pct": 0.05, "center_y_pct": 0.50, "width_pct": 0.05, "height_pct": 0.85},
        {"id": "male_header_strip_right", "type": "Connector", "center_x_pct": 0.95, "center_y_pct": 0.50, "width_pct": 0.05, "height_pct": 0.85},
        {"id": "power_led", "type": "LED", "center_x_pct": 0.24, "center_y_pct": 0.55, "width_pct": 0.04, "height_pct": 0.04}
    ]
}

STM32_BLUE_PILL_DEFAULT = {
    "board_name": "STM32 Blue Pill",
    "board_dimensions": {
        "width_mm": 53.0,
        "height_mm": 23.0,
        "pixel_width": 640,
        "pixel_height": 480
    },
    "total_critical_components": 10,
    "description": "Standard STM32 Blue Pill PCB template mapping 10 critical components.",
    "components": [
        {"id": "main_mcu_stm32", "type": "IC", "center_x_pct": 0.50, "center_y_pct": 0.45, "width_pct": 0.24, "height_pct": 0.20, "tolerance_override_mm": 1.0},
        {"id": "micro_usb_port", "type": "Connector", "center_x_pct": 0.50, "center_y_pct": 0.08, "width_pct": 0.16, "height_pct": 0.12},
        {"id": "main_crystal_8mhz", "type": "Connector", "center_x_pct": 0.50, "center_y_pct": 0.70, "width_pct": 0.12, "height_pct": 0.08},
        {"id": "rtc_crystal", "type": "Connector", "center_x_pct": 0.28, "center_y_pct": 0.72, "width_pct": 0.05, "height_pct": 0.10},
        {"id": "voltage_reg_3v3", "type": "IC", "center_x_pct": 0.26, "center_y_pct": 0.45, "width_pct": 0.08, "height_pct": 0.12},
        {"id": "reset_button", "type": "Connector", "center_x_pct": 0.25, "center_y_pct": 0.22, "width_pct": 0.08, "height_pct": 0.08},
        {"id": "boot0_header", "type": "Connector", "center_x_pct": 0.75, "center_y_pct": 0.20, "width_pct": 0.08, "height_pct": 0.06},
        {"id": "boot1_header", "type": "Connector", "center_x_pct": 0.75, "center_y_pct": 0.30, "width_pct": 0.08, "height_pct": 0.06},
        {"id": "male_header_strip_left", "type": "Connector", "center_x_pct": 0.06, "center_y_pct": 0.50, "width_pct": 0.05, "height_pct": 0.85},
        {"id": "male_header_strip_right", "type": "Connector", "center_x_pct": 0.94, "center_y_pct": 0.50, "width_pct": 0.05, "height_pct": 0.85}
    ]
}

FALLBACK_TEMPLATE = ARDUINO_UNO_DEFAULT

def load_json_template(file_path: Any) -> Dict[str, Any]:
    """
    Loads and parses a JSON PCB template file. Dynamic path checking uses absolute routing.
    If the file is missing or corrupt, it creates and saves a default layout configuration
    for that board dynamically to the templates folder on the fly.
    """
    filename = os.path.basename(str(file_path))
    stem = Path(filename).stem
    
    # Compute absolute target file path
    abs_path = os.path.join(TEMPLATES_DIR, filename)

    # Determine default schema profile based on stem name
    if "arduino_uno" in stem:
        default_layout = ARDUINO_UNO_DEFAULT
    elif "esp32_devkit" in stem:
        default_layout = ESP32_DEVKIT_DEFAULT
    elif "stm32_blue_pill" in stem:
        default_layout = STM32_BLUE_PILL_DEFAULT
    else:
        default_layout = FALLBACK_TEMPLATE

    # If file does not exist, write it dynamically on the fly
    if not os.path.exists(abs_path):
        logger.warning(f"Template not found at {abs_path}. Re-generating on-the-fly.")
        save_json_template(abs_path, default_layout)
        return default_layout
    
    try:
        with open(abs_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict):
                return data
            logger.warning(f"File data in {abs_path} is invalid. Overwriting with defaults.")
            save_json_template(abs_path, default_layout)
            return default_layout
    except Exception as e:
        logger.error(f"Error loading template config at {abs_path}: {e}. Overwriting with fallback.")
        save_json_template(abs_path, default_layout)
        return default_layout

def save_json_template(file_path: Any, data: Dict[str, Any]) -> bool:
    """
    Saves a PCB template dictionary to a formatted JSON file using absolute paths.
    """
    filename = os.path.basename(str(file_path))
    abs_path = os.path.join(TEMPLATES_DIR, filename)
    try:
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved template to: {abs_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving template to {abs_path}: {e}")
        return False

class JsonLoader:
    """
    Backward-compatible utility class proxying to standalone template loading functions.
    """
    @staticmethod
    def load(file_path: Path) -> Dict[str, Any]:
        return load_json_template(file_path)

    @staticmethod
    def save(data: Dict[str, Any], file_path: Path) -> bool:
        return save_json_template(file_path, data)
