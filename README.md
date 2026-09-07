# Automated Optical Inspection (AOI) System for PCB Component Detection & Defect Inspection

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11m-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit--Industrial-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade, industrial software infrastructure and operator console for **Automated Optical Inspection (AOI)** of Printed Circuit Board (PCB) assemblies. The system operates as a dual-pipeline inspection engine combining **PCB Component Detection & Inventory** (`DETECT → IDENTIFY → COUNT → DISPLAY → EXPORT`) with concurrent **Circuit & Solder Defect Inspection**.

---

## 📸 Presentation & Dashboard Highlights

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INDUSTRIAL PCB AOI CONTROL CONSOLE                        │
├─────────────────┬─────────────────────────┬─────────────────────────────────┤
│ TOTAL COMPONENTS│ UNIQUE COMPONENT TYPES  │ AVERAGE CONFIDENCE              │
│       45        │            9            │             74.2%               │
├─────────────────┴─────────────────────────┴─────────────────────────────────┤
│ 45 components detected across 9 component types (Avg. Conf: 74.2%)          │
├───────────────────────────┬─────────────────────────────────────────────────┤
│ COMPONENT TYPE SUMMARY    │ COMPONENT VISUALIZATION                         │
│                           │                                                 │
│ Capacitor      18         │  [ Large PCB Image with Green Bounding Boxes ]  │
│ Resistor       12         │  [ Bounding Box Labels: <class_name> <conf>  ]  │
│ IC              6         │                                                 │
│ Connector       6         │                                                 │
│ LED             2         │                                                 │
│ Switch          1         │                                                 │
│ ...                       │                                                 │
├───────────────────────────┴─────────────────────────────────────────────────┤
│ COMPONENT DETECTION DETAILS (Sorted by Confidence Descending)               │
│ #  │ Component │ Class ID │ Confidence │ Bounding Box (X1, Y1, X2, Y2) │ Center│
└────┴───────────┴──────────┴────────────┴───────────────────────────────┴───────┘
```

---

## 🚀 Key Features

- **Industrial Operator Console**: Modern dark-slate dashboard (`#0f172a`, `#1e293b`) designed for electronic assembly lines, featuring real-time state management (`IDLE`, `PROCESSING`, `COMPLETED`, `ERROR`), KPI metric cards, and responsive data tables.
- **Pure Component Detection & Inventory System**: Real-time YOLO inference detecting all physical PCB components (`resistor`, `capacitor`, `IC`, `connector`, `LED`, `diode`, `switch`, `transistor`, etc.) as the absolute source of truth without dropping or altering detections.
- **Dynamic Inventory KPI Metrics**: Computes Total Component Count, Unique Component Types, Average Confidence Percentage, and Class Summary Tables with automated sum-consistency verification ($\sum \text{Counts} == \text{Total}$).
- **Dual-Model Deep Learning Pipeline**: Concurrent routing to:
  1. **Component YOLO Model** (`Component_best.pt` / `22 classes`)
  2. **Profile-Specific Circuit Defect Models** (`DeepPCB`, `DsPCBSD+`, `HRIPCB`, `TDD-PCB`)
- **Automated Defect Model Auto-Binding**: PCB template selection automatically binds the matching defect model (e.g. Arduino Uno $\rightarrow$ DeepPCB, ESP32 $\rightarrow$ DsPCBSD+, STM32 $\rightarrow$ HRIPCB, Generic $\rightarrow$ TDD-PCB).
- **Polymorphic Exporter Factory**: Multi-format quality report generator creating:
  - **Multi-Sheet Excel Workbooks (.xlsx)**: Summary Metadata, Component Type Summary, and Full Detection Details ledger.
  - **CSV Ledgers (.csv)**: Flat tabular exports for logistics and inventory integration.
  - **PDF Quality Certificates (.pdf)**: Rendered ReportLab tables with branding and timestamps.
  - **JSON Logs (.json)**: Structured payloads for database synchronization.
- **ModelManager Caching & Hardware Acceleration**: Lazy-loading and PyTorch instance caching across Streamlit reruns, CUDA GPU acceleration support, and safe fallback weight resolution.

---

## 🛠️ Technology Stack & Requirements

- **Python Version**: `3.12` or higher
- **Deep Learning Framework**: `ultralytics` (YOLOv11)
- **Dashboard UI**: `streamlit`
- **Computer Vision & Image Processing**: `opencv-python`, `pillow`
- **Data Engineering**: `pandas`, `numpy`, `openpyxl`
- **PDF Exporter Engine**: `reportlab`
- **Configuration Management**: `pyyaml`

---

## 📂 Repository Layout

```text
pcb-quality-inspection/
├── src/
│   ├── app/
│   │   └── main.py                 # Industrial Operator Dashboard & State Machine
│   ├── ai/
│   │   ├── detection_engine.py     # YOLO Component & Circuit Inspection Engine
│   │   └── model_manager.py        # Centralized Model Manager & Caching Layer
│   ├── inspection/
│   │   ├── inspection_engine.py    # Algorithmic checker orchestrator
│   │   ├── component_counter.py    # Quantity census module
│   │   ├── missing_checker.py      # Missing component checker (Future Scope)
│   │   ├── extra_checker.py        # Unregistered item checker (Future Scope)
│   │   ├── position_checker.py     # Millimeter offset checker (Future Scope)
│   │   └── crack_checker.py        # Solder joint fracture analyzer
│   ├── mock/
│   │   └── mock_results.py         # Mock inspection payload generator
│   └── utils/
│       ├── config_loader.py        # Dynamic split YAML configuration loader
│       ├── constants.py            # Status & styling color hex codes
│       ├── file_manager.py         # Output directory manager
│       ├── helper.py               # Image converters & formatters
│       ├── json_loader.py          # JSON template loader
│       ├── logger.py               # Logging module
│       ├── report_exporter.py      # Polymorphic Report Exporter Factory (Excel/CSV/PDF/JSON)
│       ├── template_manager.py     # PCB template profile manager
│       └── validators.py           # File size & magic-byte header validators
├── configs/
│   ├── app.yaml                    # UI theme and output paths
│   ├── inference.yaml              # Confidence (0.25) & IoU (0.45) thresholds
│   ├── model.yaml                  # Model registry & profile defect mapping
│   └── training.yaml               # Training hyperparameters
├── templates/                      # PCB template profiles (arduino_uno, esp32_devkit, stm32_blue_pill, generic_pcb)
├── models/
│   └── trained/                    # Production-grade trained weights
│       ├── Component/              # Component Detector (22 classes)
│       ├── DeepPCB/                # Arduino Uno Defect Detector
│       ├── DsPCBSD+/               # ESP32 Defect Detector
│       ├── HRIPCB/                 # STM32 Defect Detector
│       └── TDD-PCB/                # Generic PCB Defect Detector
├── docs/                           # Technical documentation & presentation guides
│   ├── PROJECT_INFO.md             # In-depth system overview & presentation guide
│   ├── architecture.md             # System architecture & dataflow diagrams
│   ├── FILE_CATALOG.md             # File-by-file codebase catalog
│   └── DATASET.md                  # Custom dataset structuring guide
├── tests/
│   ├── validate_run.py             # System pipeline integration test
│   ├── test_thresholds.py          # Confidence & IoU threshold test suite
│   ├── test_workflow_scenarios.py  # Status aggregator logic test suite
│   └── diagnose_full_pipeline.py  # Reference PCB image diagnostic test
├── PROGRESS_REPORT.md              # Project progress & milestone report
└── requirements.txt                # Project dependencies
```

---

## 🤖 Deep Learning Models Architecture

The system coordinates 5 specialized YOLO models:

| Model Role | Model Target | Weights Path | Supported Classes |
| :--- | :--- | :--- | :--- |
| **Component Detector** | Common PCB Components | `models/trained/Component/Component_best.pt` | 22 classes (`battery`, `button`, `buzzer`, `capacitor`, `clock`, `connector`, `diode`, `display`, `fuse`, `heatsink`, `ic`, `inductor`, `led`, `pads`, `pins`, `potentiometer`, `relay`, `resistor`, `switch`, `transducer`, `transformer`, `transistor`) |
| **Arduino Defect Detector** | `arduino_uno` PCB | `models/trained/DeepPCB/DeepPCB.pt` | 6 classes (`open`, `short`, `mousebite`, `spur`, `copper`, `pin_hole`) |
| **ESP32 Defect Detector** | `esp32_devkit` PCB | `models/trained/DsPCBSD+/DsPCBSD+.pt` | 9 classes (`SH`, `SP`, `SC`, `OP`, `MB`, `HB`, `CS`, `CFO`, `BMFO`) |
| **STM32 Defect Detector** | `stm32_blue_pill` PCB | `models/trained/HRIPCB/HRIPCB.pt` | 6 classes (`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`) |
| **Generic Defect Detector** | `generic_pcb` PCB | `models/trained/TDD-PCB/PDD-PCB-best.pt` | 6 classes (`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`) |

---

## 💻 Installation & Quickstart

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kaushikajani3002-rgb/pcb-quality-inspection.git
   cd pcb-quality-inspection
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run System Integration Verification**:
   ```bash
   python tests/validate_run.py
   ```

5. **Launch Industrial Dashboard Console**:
   ```bash
   streamlit run src/app/main.py
   ```
   Open browser at: **`http://localhost:8501`**

---

## 📊 Exported Reports Structure

- **Excel (`.xlsx`)**:
  - `Summary Sheet`: Board Name, Image Name, Timestamp, Model Path, Total Detections, Unique Types, Average Confidence, Confidence/IoU Thresholds, Hardware Device.
  - `Component Summary Sheet`: Component Class vs Count.
  - `Detection Details Sheet`: Inventory ledger (`#`, `Image`, `Component Class`, `Class ID`, `Confidence`, `X1`, `Y1`, `X2`, `Y2`, `Center X %`, `Center Y %`).
- **CSV (`.csv`)**: Flat tabular file for database/ERP ingestion.
- **PDF (`.pdf`)**: Formatted quality certificate with summary metrics and inventory details.
- **JSON (`.json`)**: Machine-readable inspection log.

---

## 🧪 Verification & Test Suite

Run all automated verification tests:
```bash
python tests/validate_run.py
python tests/test_thresholds.py
python -m unittest tests/test_workflow_scenarios.py
python tests/diagnose_full_pipeline.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 👤 Project Team & Authors

- **Kaushik Ajani**
- **Vidhi Ranpura**
- **Isha Kakadiya**
- **Tushar Kacha**
- **GitHub Repository**: [kaushikajani3002-rgb/pcb-quality-inspection](https://github.com/kaushikajani3002-rgb/pcb-quality-inspection)

