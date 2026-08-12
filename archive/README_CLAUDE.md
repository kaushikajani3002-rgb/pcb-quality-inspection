# Developer Handover & System Architecture Guide (Tailored for Claude AI)

This document provides a complete technical map of the **AI-Based Automated Optical Inspection (AOI) System for PCB Assembly Verification** codebase. It outlines the architecture, file specs, coordinate systems, implemented features, and the step-by-step integration roadmap.

---

## 1. Project Context & Objectives
- **Goal**: Build a robust industrial software infrastructure wrapping Automated Optical Inspection checks for PCB assemblies.
- **AI/ML Constraint**: All computer vision, deep learning (YOLO11m/YOLO11m-Seg), and image processing models are decoupled. The current system relies on a **Mock Data Layer** that simulates predictions, annotations, and overlays.
- **Industrial Standards**: High-fidelity dark-themed operator console, physical millimeter-based alignment checks, exception-safe validation pipelines, and polymorphic CSV/JSON/PDF reporting.

---

## 2. Directory Structure & File Catalog
Below is the directory footprint implemented inside the `PCB_AOI/` directory:

```
PCB_AOI/
├── app/
│   ├── __init__.py
│   └── main.py                     # Streamlit Operator UI & Session State Machine
├── config/
│   └── config.yaml                 # System configurations, thresholds, and paths
├── templates/
│   ├── arduino_uno.json            # Layout config: Arduino Uno (14 components)
│   ├── esp32_devkit.json           # Layout config: ESP32 DevKit (9 components)
│   └── stm32_blue_pill.json        # Layout config: STM32 Blue Pill (10 components)
├── mock/
│   ├── __init__.py
│   └── mock_results.py             # Image drawer (Pillow) & simulated inspection responses
├── inspection/
│   ├── __init__.py
│   ├── inspection_engine.py        # Central check orchestrator
│   ├── component_counter.py        # quantity discrepancy checks
│   ├── missing_checker.py          # ID-based missing component scanner
│   ├── extra_checker.py            # Highlights unregistered objects
│   ├── position_checker.py         # Millimeter-based Euclidean distance verification
│   └── crack_checker.py            # Solder joint fracture indicators
├── utils/
│   ├── __init__.py
│   ├── config_loader.py            # Dot-notation YAML configurations loader
│   ├── json_loader.py              # Exception-safe JSON loader/writer (relative path resolver)
│   ├── template_manager.py         # Validates template schemas (board_name, width_mm)
│   ├── report_exporter.py          # Factory Pattern Exporter for PDF (ReportLab), CSV, JSON
│   ├── file_manager.py             # Ensures logs/reports/outputs directories exist
│   ├── validators.py               # PNG/JPG file size & magic byte signature validators
│   ├── constants.py                # Status constants and industrial CSS colors
│   └── helper.py                   # Formatting utilities (duration, time, image bytes)
├── logs/
│   └── inspection.log              # Master system trace log
├── reports/                        # Saved inspection logs (.pdf, .csv, .json)
├── outputs/                        # Stored analysis camera frames
├── requirements.txt                # Package dependencies
├── README.md                       # Operations guide
├── project_summary.txt             # Text-formatted implementation log
└── README_CLAUDE.md                # [THIS FILE] Context ingest guide
```

---

## 3. Class & Method Specifications

### A. Core Engine (`inspection/`)
- `InspectionEngine` (`inspection_engine.py`):
  - `inspect_board(template, detected_components, position_tolerance, precalculated_cracks) -> Dict[str, Any]`
  - Aggregates outputs from the component counter, missing, extra, alignment, and crack checkers. Returns overall `PASS` or `FAIL`.
- `PositionChecker` (`position_checker.py`):
  - `check(expected, detected, default_tolerance_mm, board_dimensions) -> List[Dict[str, Any]]`
  - Performs physical coordinate misalignment checking.

### B. Mock Layer (`mock/`)
- `MockInspectionService` (`mock_results.py`):
  - `run_inspection(template, defect_mode, confidence_threshold, iou_threshold, position_tolerance) -> Dict[str, Any]`
    - Generates passing status or introduces mock discrepancies (missing chip, shifted regulator, crack joint, rogue capacitor).
  - `generate_mock_images(template, results) -> Tuple[Image, Image, Image]`
    - Dynamically draws (1) Clean grey PCB base, (2) Bounding Box annotations, and (3) Semi-transparent segmentation masks with crack indicators using `Pillow`.

### C. Utilities (`utils/`)
- `JsonLoader` (`json_loader.py`):
  - `load_json_template(file_path) -> Dict[str, Any]`
  - `save_json_template(file_path, data) -> bool`
  - Safe reader/writer. If templates are missing or corrupted, **automatically reconstructs the default board configuration files on disk** to prevent crashes. Resolves base paths relative to root directory.
- `TemplateManager` (`template_manager.py`):
  - `load_template(template_name) -> Dict[str, Any]`
  - `validate_template(template_data) -> bool`
  - Validates keys: `board_name`, `board_dimensions`, `total_critical_components`, `components`.
- `ReportExporterFactory` (`report_exporter.py`):
  - Implements Factory Pattern for `BaseReportExporter`. Concrete classes: `JSONReportExporter`, `CSVReportExporter`, `PDFReportExporter` (using ReportLab document tables and layout paragraphs).
- `ImageValidator` & `ConfigurationValidator` (`validators.py`):
  - `validate_file_metadata(filename, size)`: Checks extensions and file boundaries.
  - `validate_image_bytes(image_bytes)`: Checks magic byte headers (`b"\x89PNG\r\n\x1a\n"`, `b"\xff\xd8\xff"`) to block spoofed attachments.
  - `validate_inspection_parameters(confidence, iou, position_tolerance)`: Validates sliders.

---

## 4. Coordinate & Math Transformations
To allow the system to operate independently of canvas sizes and hardware resolutions, it employs **Normalized Coordinate Percentages (0.0 to 1.0)** relative to the board boundaries:

1. **Pixel Coordinates for Visual Rendering**:
   When rendering components onto a canvas of size ($W_{\text{pixel}} \times H_{\text{pixel}}$):
   $$x_{\text{pixel}} = x_{\text{pct}} \times W_{\text{pixel}}$$
   $$y_{\text{pixel}} = y_{\text{pct}} \times H_{\text{pixel}}$$
   $$\text{width}_{\text{pixel}} = \text{width}_{\text{pct}} \times W_{\text{pixel}}$$
   $$\text{height}_{\text{pixel}} = \text{height}_{\text{pct}} \times H_{\text{pixel}}$$

2. **Euclidean Misalignment Verification (Millimeters)**:
   Instead of testing raw pixels, we calculate structural offsets in physical millimeters:
   $$\Delta x_{\text{mm}} = (x_{\text{actual, pct}} - x_{\text{expected, pct}}) \times \text{board\_width\_mm}$$
   $$\Delta y_{\text{mm}} = (y_{\text{actual, pct}} - y_{\text{expected, pct}}) \times \text{board\_height\_mm}$$
   $$\text{Offset}_{\text{mm}} = \sqrt{(\Delta x_{\text{mm}})^2 + (\Delta y_{\text{mm}})^2}$$

   If $\text{Offset}_{\text{mm}} > \text{Tolerance}_{\text{mm}}$, the component is flagged as `MISALIGNED`.

---

## 5. UI Features & Caching Strategy
- **Master UI (`app/main.py`)**: Runs a session state machine (`st.session_state.workflow_status`). Displays dynamic info boxes containing name, dimensions (mm), expected counts, and lists of components for the active selection.
- **Reloading Mechanism**: To prevent Streamlit from caching outdated modules in memory (which causes systemic file-loading and validation issues when helper modules are updated), the entrypoint script imports `importlib` and force-reloads all helper modules from disk on rerun:
  ```python
  import importlib
  for mod in ["utils.json_loader", "utils.template_manager", "utils.logger", "mock.mock_results", "inspection.inspection_engine"]:
      if mod in sys.modules:
          importlib.reload(sys.modules[mod])
  ```

---

## 6. Guide for Claude: Integrating YOLO & OpenCV
When you (Claude AI) take over to implement the visual computer vision pipeline:

1. **Step 1: Image Alignment (OpenCV)**:
   - Create `inspection/preprocessor.py`.
   - Ingest the raw board image from the operator dashboard camera.
   - Load expected dimensions from `template["board_dimensions"]`.
   - Use keypoint detectors (ORB/SIFT) or fiducial markers to calculate the homography matrix:
     `H, status = cv2.findHomography(actual_pts, template_pts)`
   - Apply perspective warping to align the incoming frame to the exact template dimensions:
     `aligned_img = cv2.warpPerspective(src_img, H, (pixel_width, pixel_height))`

2. **Step 2: Model Inference (YOLO11m)**:
   - Create a model wrapper class in `models/yolo_wrapper.py` loading `yolo11m.pt` (object detection) and `yolo11m-seg.pt` (solder segmentations).
   - Feed the aligned image into the model:
     `results = model(aligned_img, conf=confidence_threshold, iou=iou_threshold)`
   - Convert YOLO coordinates to normalized percentages:
     `center_x_pct = box.xywhn[0]`
     `center_y_pct = box.xywhn[1]`
   - Package these detections into a list matching the `detected_components` format:
     `[{"id": "detected_part_01", "type": "IC", "center_x_pct": 0.50, ...}]`

3. **Step 3: Connect to Inspection Engine**:
   - Query the `InspectionEngine().inspect_board()` with the template and the YOLO-derived coordinate lists.
   - Pass the output to the existing `report_exporter.py` classes and dashboard tables.
