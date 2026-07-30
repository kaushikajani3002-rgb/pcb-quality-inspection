# AI-Based Automated Optical Inspection (AOI) System for PCB Assembly Verification

A production-grade, modular software engineering infrastructure designed to wrap PCB assembly validation models. This system features template matching, error validation pipelines, structured logging, a polymorphic report exporter (PDF, CSV, JSON), and an industrial Streamlit operator dashboard.

## Project Overview
In electronics manufacturing, Automated Optical Inspection (AOI) is crucial for identifying assembly errors (missing ICs, misaligned resistors, extra solder bridges, and solder joint cracks). While the AI models (YOLO11m, YOLO11m-Seg) handle visual predictions, this repository provides the high-performance application infrastructure, state machine, logging, configuration, and verification logic necessary for deployment.

---

## Software Architecture
The application is structured following the **Clean Architecture** paradigm:
1. **Core Domain (Inspection Engine)**: Located in `inspection/`. Contains pure checker algorithms (counting, missing items, misalignment calculations, cracks) with zero external dashboard dependencies.
2. **Utilities**: Located in `utils/`. Reusable, single-responsibility wrappers for configurations, templated JSON, structured logging, input validations, and ReportLab PDF document layout flowables.
3. **Data Mocking Layer**: Located in `mock/`. Simulates realistic YOLO bounding box coordinates, segmentation masks, and pixel arrays for PASS/FAIL scenarios, rendering the application immediately testable.
4. **Presentation (Dashboard UI)**: Located in `app/main.py`. A Streamlit-based web interface operating as a state machine.

---

## Folder Structure
```
PCB_AOI/
├── app/
│   ├── __init__.py
│   └── main.py                   # Streamlit Entry point & State Machine
├── config/
│   └── config.yaml               # Inspection thresholds, tolerances, & path definitions
├── templates/
│   ├── arduino_uno.json          # Expected component layout template for Arduino Uno
│   └── esp32.json                # Expected component layout template for ESP32
├── mock/
│   ├── __init__.py
│   └── mock_results.py           # Simulated model predictions & Pillow image renders
├── utils/
│   ├── __init__.py
│   ├── config_loader.py          # ConfigLoader class (dot notation retrieval)
│   ├── json_loader.py            # Exception-safe JSON loader/writer
│   ├── logger.py                 # File-based logging to logs/inspection.log
│   ├── report_exporter.py        # Abstract Base Exporter & Factory for JSON/CSV/PDF
│   ├── file_manager.py           # Pre-run path creator
│   ├── validators.py             # Magic-number byte headers & configuration boundaries
│   ├── constants.py              # Operational statuses, industrial HEX color codes
│   └── helper.py                 # Time, date, and byte converters
├── inspection/
│   ├── __init__.py
│   ├── inspection_engine.py      # Aggregates individual verification checkers
│   ├── component_counter.py      # Quantity variance calculator
│   ├── missing_checker.py        # Spatial-to-template coordinate comparison
│   ├── extra_checker.py          # Unexpected component locator
│   ├── position_checker.py       # Euclidean distance misalignment validator
│   └── crack_checker.py          # Joint fracture boundary detector
├── reports/                      # Exporter target output directory (.pdf, .csv, .json)
├── outputs/                      # Camera and analysis annotation image outputs
├── logs/                         # File logs target directory (inspection.log)
├── assets/                       # Image assets and logo placeholders
├── docs/
│   └── architecture.md           # System architecture & future integration instructions
├── requirements.txt              # Standard library extensions
└── README.md                     # This document
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- `pip` (Python package manager)

### Step 1: Clone and Navigate
```bash
cd d:/PCB/PCB_AOI
```

### Step 2: Set up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows PowerShell
.\venv\Scripts\Activate.ps1
# On Linux / macOS
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application
Launch the Streamlit operator dashboard locally:
```bash
streamlit run app/main.py
```
Open the browser at: `http://localhost:8501`

---

## Dashboard Features & Interface Guide
1. **Logo & Terminal Panel**: Identifies active operator ID and system timestamp.
2. **Template Selection**: Dynamically scans the `templates/` directory for profiles (`Arduino Uno Rev3` / `ESP32 DevKitC`).
3. **Parameter Sliders**: Adjust Confidence levels, IoU thresholds, and position tolerance boundaries (in pixels).
4. **Defect Toggler (Simulation)**: Check "Force Anomaly/Defect Mode" to switch output.
   - **Unchecked (PASS)**: Displays a green PASS banner; component lists align perfectly.
   - **Checked (FAIL)**: Simulates a board containing a missing MCU chip, a misaligned regulator, an extra capacitor, and a joint crack.
5. **Tabs**: Switch between:
   - **Original Camera View**: Displays clean grey PCB and components.
   - **YOLO BBoxes**: Draws green (correct), yellow (misaligned), red crosshatch (missing), and pink (extra) borders.
   - **YOLO-Seg Crack Overlay**: Fills semitransparent masks and draws blue solder cracks.
6. **Ledgers**: Lists individual inventories and lists discrepancies in a table.
7. **Report Downloading**: Generates on-the-fly PDF certificates, CSV registers, and JSON database logs.

---

## System Configurations (`config/config.yaml`)
Customize parameters without touching code:
```yaml
inspection:
  confidence: 0.50
  iou: 0.45
  position_tolerance: 15.0  # Euclidean pixel tolerance
paths:
  template_folder: "templates"
  report_folder: "reports"
  log_folder: "logs"
```

---

## Report Generator & File Export Factory
Refactored using a Polymorphic Factory design pattern:
- **Base class**: `BaseReportExporter`
- **JSON**: Writes raw dictionary state.
- **CSV**: Generates clean columns suitable for Excel or downstream logistics dashboards.
- **PDF**: Employs ReportLab flowables to build multi-page tables, headers, and color-coded status blocks.
