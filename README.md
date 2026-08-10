# Automated Optical Inspection (AOI) System for PCB Assembly Verification

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11m-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade software infrastructure and operator console for **Automated Optical Inspection (AOI)** of populated Printed Circuit Board (PCB) assemblies. This application wraps deep learning (YOLO11m component detection) and custom verification checkers to validate physical alignment offsets (in millimeters), quantities, missing components, and solder joint cracks against reference layouts.

---

## 🚀 Key Features

- **Industrial Operator UI**: Modern, dark-themed Streamlit control console designed for assembly line operators.
- **YOLO11m Component Detection**: Real-time component bounding box inference and quantity census.
- **Physical Millimeter Verification**: Euclidean distance position checkers calculating component displacements in physical millimeters ($\text{mm}$) using normalized percentages relative to template boundary dimensions.
- **Solder Joint Inspection**: Segmentation overlays representing solder joint statuses and highlighting joint crack fractures.
- **Polymorphic Exporters**: Factory Pattern report generator creating certified quality outputs in **PDF** (rendered tables and styling), **CSV** (ledger reports), and **JSON** (database synchronization logs).
- **Graceful Hardware Fallback**: Safe loading blocks verifying file integrity, size boundaries, magic-byte header signatures, and weights configurations.

---

## 🛠️ Technology Stack & Requirements

- **Python Version**: `3.12` or higher
- **Core Packages**:
  - `ultralytics` (YOLOv11 Deep Learning Models)
  - `streamlit` (Operator Dashboard)
  - `opencv-python` (Perspective warping and alignment)
  - `pillow` (Overlay drawing and image rendering)
  - `reportlab` (PDF generation factory)
  - `pandas` & `numpy` (Inventory statistics and matrix math)
  - `pyyaml` (System configuration loaders)

---

## 📂 Repository Directory Layout

```text
PCB_AOI/
├── app/
│   ├── __init__.py
│   └── main.py                     # Streamlit Operator UI & State Machine
├── config/
│   └── config.yaml                 # Inspection tolerances, paths, & default configurations
├── docs/
│   └── architecture.md             # In-depth architectural designs
├── inspection/
│   ├── __init__.py
│   ├── inspection_engine.py        # Master checker orchestrator
│   ├── component_counter.py        # Counts detections against expected totals
│   ├── missing_checker.py          # Finds absent expected components
│   ├── extra_checker.py            # Highlights unregistered objects
│   ├── position_checker.py         # Performs millimeter-based displacement calculations
│   └── crack_checker.py            # Detects solder joint fractures
├── mock/
│   ├── __init__.py
│   └── mock_results.py             # Generates mock inspection data
├── templates/
│   ├── arduino_uno.json            # Layout dimensions/coordinates: Arduino Uno
│   ├── esp32_devkit.json           # Layout dimensions/coordinates: ESP32 DevKit
│   └── stm32_blue_pill.json        # Layout dimensions/coordinates: STM32 Blue Pill
├── utils/
│   ├── __init__.py
│   ├── config_loader.py            # Loads configuration options
│   ├── constants.py                # Status and styling hex colors
│   ├── file_manager.py             # Directory builder
│   ├── helper.py                   # Formatters and converters
│   ├── json_loader.py              # Safe JSON template reader/writer
│   ├── logger.py                   # File logger configuration
│   ├── report_exporter.py          # Abstract report exporter classes (Factory Pattern)
│   ├── template_manager.py         # Manages template updates and checks
│   └── validators.py               # Size & magic-byte header validators
├── DATASET.md                      # Instructions on dataset structures & custom data
├── PROJECT_INFO.md                 # Detailed onboarding & project walkthrough
├── detection_engine.py             # YOLO Inference wrapper & coordination module
├── train_component_yolo.py         # YOLO11m component detection training script
├── finetune_defect_yolo.py         # YOLO11m trace defect fine-tuning script
├── inference_defect_yolo.py        # Batch inference defect analyzer
└── requirements.txt                # Project dependencies
```

---

## 💻 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kaushikajani3002-rgb/pcb-defect-detection-yolo.git
   cd pcb-defect-detection-yolo
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

4. **Add Trained Weights**:
   Place your trained model weights inside the `models/` directory (create this directory if it doesn't exist):
   - Component Detection: `models/component_yolo11m.pt`
   - Defect Detection: `models/dspcbsd.pt`, `models/deeppcb.pt`, etc.
   
   Configure the paths inside the `MODEL_PATHS` dictionary in `detection_engine.py`.

---

## ⚡ Running the Dashboard Console

Launch the Streamlit operator dashboard locally:
```bash
streamlit run app/main.py
```
Open your browser at: **`http://localhost:8501`**

---

## 🏋️ Training & Fine-Tuning Models

See [DATASET.md](DATASET.md) for how to structure your custom PCB images.

### Train Component Detector
To train the YOLO11m component detection model to recognize parts:
```bash
python train_component_yolo.py --data datasets/components/data.yaml --epochs 70 --batch 8
```

### Fine-Tune Defect Detector
To fine-tune pre-trained component weights on defect datasets (e.g. solder joint issues or trace damage):
```bash
python finetune_defect_yolo.py
```

### Run Batch Defect Inference
To execute batch inference on a folder of test PCB frames:
```bash
python inference_defect_yolo.py
```

---

## 🔧 Inspection Configuration (`config/config.yaml`)

You can customize parameters directly without changing the codebase:
```yaml
inspection:
  confidence: 0.50            # YOLO confidence threshold
  iou: 0.45                   # Non-Maximum Suppression (NMS) IoU limit
  position_tolerance: 15.0    # Default position checker boundary limit
```

---

## 📈 Planned Work / Future Integration

1. **OpenCV Perspective Preprocessor**: Implement homography transformation to align incoming frames automatically using board corner fiducials.
2. **Defect Model Integration**: Move remaining defect checking tabs from mock generators to active deep learning inference once segmentation models are trained.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

- **Kaushik Ajani**
- **Vidhi Rampura**
- **Isha Kakadiya**
- **Tushar Kacha**
- **GitHub**: [kaushikajani3002-rgb](https://github.com/kaushikajani3002-rgb)
- 
