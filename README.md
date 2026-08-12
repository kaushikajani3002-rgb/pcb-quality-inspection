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

> [!NOTE]
> For an in-depth class-by-class and file-by-file description of every folder and script in this codebase, refer to the [Directory & File Catalog](docs/FILE_CATALOG.md).

```text
pcb-quality-inspection/
├── src/
│   ├── app/
│   │   └── main.py                 # Streamlit Operator UI & State Machine
│   ├── ai/
│   │   └── detection_engine.py     # YOLO Inference wrapper & coordination module
│   ├── inspection/
│   │   ├── inspection_engine.py    # Master checker orchestrator
│   │   ├── component_counter.py    # Counts detections against expected totals
│   │   ├── missing_checker.py      # Finds absent expected components
│   │   ├── extra_checker.py        # Highlights unregistered objects
│   │   ├── position_checker.py     # Millimeter-based displacement calculations
│   │   └── crack_checker.py        # Detects solder joint fractures
│   ├── mock/
│   │   └── mock_results.py         # Generates mock inspection data
│   └── utils/
│       ├── config_loader.py        # Dynamic split configuration loader
│       ├── constants.py            # Status and styling hex colors
│       ├── file_manager.py         # System directory initializer
│       ├── helper.py               # Formatters and Pillow image converters
│       ├── json_loader.py          # Safe JSON template reader/writer
│       ├── logger.py               # File logger configuration
│       ├── report_exporter.py      # Abstract report exporter classes (Factory Pattern)
│       ├── template_manager.py     # Manages template updates and checks
│       └── validators.py           # Size & magic-byte header validators
├── configs/
│   ├── app.yaml                    # UI and report path configurations
│   ├── inference.yaml              # Confidence and alignment tolerances
│   ├── model.yaml                  # Trained and pre-trained model paths
│   └── training.yaml               # Training hyperparameters
├── training/
│   ├── component_detection/        # YOLO component detection training pipeline
│   │   └── train_component_yolo.py
│   └── defect_detection/           # YOLO defect fine-tuning pipeline
│       └── finetune_defect_yolo.py
├── inference/
│   └── inference_defect_yolo.py    # Standalone batch inference evaluator
├── datasets/                       # Datasets policy & documentation
│   ├── external/
│   │   └── ALL_cercit/             # Moved dataset files
│   └── README.md
├── models/
│   ├── pretrained/                 # COCO pre-trained base model weights
│   ├── trained/                    # Production-grade fine-tuned models
│   ├── checkpoints/                # Epoch weights checkpoints
│   ├── exported/                   # ONNX/TensorRT deployments
│   └── registry.yaml               # Tracks registry details of models
├── templates/                      # PCB template profiles (arduino, esp32...)
├── docs/                           # Documentation and guides
│   ├── DATASET.md                  # Custom dataset structuring guide
│   └── PROJECT_INFO.md             # Onboarding project walkthrough
├── tests/
│   └── validate_run.py             # Pipeline integration test
└── requirements.txt                # Project dependencies
```

---

## 💻 Installation & Setup

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

4. **Verify Weight Setup**:
   Trained weights are registered inside `models/trained/component_detector_best.pt` and loaded automatically by `src/ai/detection_engine.py` when running inference. Pretrained YOLO weights live in `models/pretrained/`.

---

## ⚡ Running the Dashboard Console

Launch the Streamlit operator dashboard locally:
```bash
streamlit run src/app/main.py
```
Open your browser at: **`http://localhost:8501`**

---

## 🏋️ Training & Fine-Tuning Models

See [docs/DATASET.md](docs/DATASET.md) for how to structure your custom PCB images.

### Train Component Detector
To train the YOLO11m component detection model to recognize parts:
```bash
python training/component_detection/train_component_yolo.py --data datasets/components/data.yaml --epochs 70 --batch 8
```

### Fine-Tune Defect Detector
To fine-tune pre-trained component weights on defect datasets:
```bash
python training/defect_detection/finetune_defect_yolo.py
```

### Run Standalone Batch Inference
To evaluate predictions over test images:
```bash
python inference/inference_defect_yolo.py
```

---

## 🔧 Inspection Configuration (`configs/inference.yaml`)

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
- **Vidhi Ranpura**
- **Isha Kakadiya**
- **Tushar Kacha**
- **GitHub**: [kaushikajani3002-rgb](https://github.com/kaushikajani3002-rgb)
