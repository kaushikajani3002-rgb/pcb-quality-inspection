# Project Directory & File Catalog

This catalog provides an in-depth breakdown of every folder and file in the **PCB Quality Inspection (AOI)** repository. Use this document to understand the codebase structure, locate specific files, and see what code lies inside each script.

---

## 📂 Summary of Project Structure

```text
pcb-quality-inspection/
├── configs/                        # YAML settings sheets
├── docs/                           # Documentation guides
├── src/                            # Main application source code
│   ├── ai/                         # Deep learning inference
│   ├── app/                        # Streamlit operator dashboard
│   ├── inspection/                 # Core alignment and defect checkers
│   ├── mock/                       # Data generation simulators
│   └── utils/                      # Helper libraries and report exporters
├── templates/                      # PCB reference coordinate profiles (JSON)
├── training/                       # YOLO model training scripts
├── inference/                      # Standalone batch inference script
├── tests/                          # Automated integration tests
└── requirements.txt                # Python package list
```

---

## ⚙️ 1. Configurations Folder (`configs/`)

These YAML files contain all settings, thresholds, and weights locations.

* **[configs/app.yaml](file:///d:/PCB/PCB_AOI/configs/app.yaml)**
  - *Purpose*: Configures user interface parameters and file paths.
  - *Inside*: Theme variables (`Theme: dark`), default operator profiles, output folders for quality reports, and directory paths for saved model logs.
* **[configs/inference.yaml](file:///d:/PCB/PCB_AOI/configs/inference.yaml)**
  - *Purpose*: Configures deep learning inference parameters and alignment thresholds.
  - *Inside*: `confidence_threshold` (0.50), non-maximum suppression `iou_threshold` (0.45), and default physical alignment tolerance in millimeters (`position_tolerance: 15.0 mm`).
* **[configs/model.yaml](file:///d:/PCB/PCB_AOI/configs/model.yaml)**
  - *Purpose*: Maps friendly model IDs to physical `.pt` model weights file paths.
  - *Inside*: Binds key models like `Component` (`models/trained/component_detector_best.pt`), `DeepPCB`, `DsPCBSD+`, `HRIPCB`, and `TDD-PCB`.
* **[configs/training.yaml](file:///d:/PCB/PCB_AOI/configs/training.yaml)**
  - *Purpose*: Setup hyperparameter configurations for model training.
  - *Inside*: Default learning rates (`lr0: 0.01`), epochs (70), batch sizes (8), and optimizer algorithms (`optimizer: auto`).

---

## 🖥️ 2. Main Source Code Folder (`src/`)

Contains the core application logic divided into specific modules.

### A. Deep Learning Inference Engine (`src/ai/`)
* **[src/ai/detection_engine.py](file:///d:/PCB/PCB_AOI/src/ai/detection_engine.py)**
  - *Purpose*: Loads YOLO models, runs object inference on uploaded PCB images, and matches detections back to layout templates.
  - *Functions*:
    - `load_model(model_name)`: Loads YOLO weight files using the configurations in `configs/model.yaml` with exception fallbacks.
    - `run_component_counting(...)`: Runs YOLO inference, extracts bounding box coordinates, converts pixel measurements to physical millimeters, and performs alignment mapping.
    - `match_detections_to_template(...)`: Implements a greedy matching matrix that pairs YOLO-detected boxes with template-expected components using a proximity-based Euclidean metric.

### B. Dashboard Frontend Console (`src/app/`)
* **[src/app/main.py](file:///d:/PCB/PCB_AOI/src/app/main.py)**
  - *Purpose*: Streamlit frontend script. It handles the application state machine (`IDLE`, `PROCESSING`, `COMPLETED`, `ERROR`), renders sidebar selectors, allows file uploads, draws visual results tabs, and coordinates report exporting.

### C. Inspection Verification Engine (`src/inspection/`)
* **[src/inspection/inspection_engine.py](file:///d:/PCB/PCB_AOI/src/app/inspection/inspection_engine.py)**
  - *Purpose*: The master orchestrator that receives raw YOLO detections and chains together the individual defect checkers.
  - *Class*: `InspectionEngine`
    - Runs the component counter, missing components checker, unregistered (extra) components checker, and millimeter displacement checker. Returns an aggregated verification record.
* **[src/inspection/component_counter.py](file:///d:/PCB/PCB_AOI/src/app/inspection/component_counter.py)**
  - *Purpose*: Counts total expected vs. actual placed components of each component type.
* **[src/inspection/missing_checker.py](file:///d:/PCB/PCB_AOI/src/app/inspection/missing_checker.py)**
  - *Purpose*: Scans expected template components and flags those that had no nearby YOLO detections matching their type.
* **[src/inspection/extra_checker.py](file:///d:/PCB/PCB_AOI/src/app/inspection/extra_checker.py)**
  - *Purpose*: Finds YOLO detections that are too far from any expected template coordinate, marking them as unregistered/extra items.
* **[src/inspection/position_checker.py](file:///d:/PCB/PCB_AOI/src/app/inspection/position_checker.py)**
  - *Purpose*: Computes the distance between expected coordinate centers and YOLO actual centers. Compares the offset against physical alignment tolerances in physical millimeters ($\text{mm}$).
* **[src/inspection/crack_checker.py](file:///d:/PCB/PCB_AOI/src/app/inspection/crack_checker.py)**
  - *Purpose*: Flags solder joints cracks.

### D. Mock Simulator (`src/mock/`)
* **[src/mock/mock_results.py](file:///d:/PCB/PCB_AOI/src/mock/mock_results.py)**
  - *Purpose*: Generates mock validation results and image canvases (bounding boxes, solder joint segmentation masks) for fallback runs if YOLO models are missing.

### E. Utilities & Exporters (`src/utils/`)
* **[src/utils/config_loader.py](file:///d:/PCB/PCB_AOI/src/utils/config_loader.py)**
  - *Purpose*: Loads, parses, and merges split YAML sheets in the `configs/` folder.
* **[src/utils/constants.py](file:///d:/PCB/PCB_AOI/src/utils/constants.py)**
  - *Purpose*: Declares global application constants (e.g. state strings, success/failure hex colors).
* **[src/utils/file_manager.py](file:///d:/PCB/PCB_AOI/src/utils/file_manager.py)**
  - *Purpose*: System directory helper ensuring local folders (`outputs/`, `logs/`, `reports/`) exist.
* **[src/utils/helper.py](file:///d:/PCB/PCB_AOI/src/utils/helper.py)**
  - *Purpose*: Measurement converters (pixels to physical millimeters) and Pillow-to-OpenCV typecast scripts.
* **[src/utils/json_loader.py](file:///d:/PCB/PCB_AOI/src/utils/json_loader.py)**
  - *Purpose*: Exception-safe loader that reads template profiles, validates keys, and writes fallback structures.
* **[src/utils/logger.py](file:///d:/PCB/PCB_AOI/src/utils/logger.py)**
  - *Purpose*: Configures file logging to write active traces to `logs/inspection.log`.
* **[src/utils/report_exporter.py](file:///d:/PCB/PCB_AOI/src/utils/report_exporter.py)**
  - *Purpose*: Implements the Factory Pattern to compile and export inspection logs into **JSON** (for databases), **CSV** (for tables), and **PDF** (rendered quality certificates using ReportLab).
* **[src/utils/template_manager.py](file:///d:/PCB/PCB_AOI/src/utils/template_manager.py)**
  - *Purpose*: Reads and indexes board profiles located in the `templates/` folder.
* **[src/utils/validators.py](file:///d:/PCB/PCB_AOI/src/utils/validators.py)**
  - *Purpose*: Checks uploaded images for size limits, file types, and magic-byte header signatures.

---

## 🗂️ 3. Board Profiles (`templates/`)

Reference layout configurations with physical millimeter dimensions and component coordinates:

* **[templates/arduino_uno.json](file:///d:/PCB/PCB_AOI/templates/arduino_uno.json)**: Reference layout for Arduino Uno (14 components).
* **[templates/esp32_devkit.json](file:///d:/PCB/PCB_AOI/templates/esp32_devkit.json)**: Reference layout for ESP32 DevKit (9 components).
* **[templates/stm32_blue_pill.json](file:///d:/PCB/PCB_AOI/templates/stm32_blue_pill.json)**: Reference layout for STM32 Blue Pill (10 components).

---

## 🏋️ 4. Model Training & Pipelines

Scripts used to train and fine-tune models:

* **[training/component_detection/train_component_yolo.py](file:///d:/PCB/PCB_AOI/training/component_detection/train_component_yolo.py)**
  - *Purpose*: Trains or fine-tunes the YOLO11m component detection model.
* **[training/defect_detection/finetune_defect_yolo.py](file:///d:/PCB/PCB_AOI/training/defect_detection/finetune_defect_yolo.py)**
  - *Purpose*: Fine-tunes pre-trained component models on trace or solder defect datasets.
* **[inference/inference_defect_yolo.py](file:///d:/PCB/PCB_AOI/inference/inference_defect_yolo.py)**
  - *Purpose*: Runs batch inference on directories of raw PCB images and writes text-based classifications.

---

## 🧪 5. Verification Scripts (`tests/`)

* **[tests/validate_run.py](file:///d:/PCB/PCB_AOI/tests/validate_run.py)**
  - *Purpose*: Integration test that runs when executed, validating imports, configuration loaders, template profiles, and simulated inspections.
