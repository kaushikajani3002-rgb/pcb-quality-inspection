# PCB AOI System-wide constants

# Operational statuses
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

# State Machine workflow statuses
STATE_IDLE = "IDLE"
STATE_PROCESSING = "PROCESSING"
STATE_COMPLETED = "COMPLETED"
STATE_ERROR = "ERROR"

# Dashboard Industrial Hex Colors
COLOR_PASS = "#00FF66"      # Bright Industrial Green
COLOR_FAIL = "#FF3333"      # Bright Industrial Red
COLOR_WARNING = "#FFCC00"   # Warning Amber
COLOR_INFO = "#00CCFF"      # Cyan Info
COLOR_CRACK = "#3399FF"     # Light Blue for Cracks

# Component Categories
COMPONENT_TYPES = ["IC", "Resistor", "Capacitor", "LED", "Diode", "Connector"]

# Default inspection criteria bounds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_IOU = 0.0
MAX_IOU = 1.0
