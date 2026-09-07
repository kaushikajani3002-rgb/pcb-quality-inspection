# Technical System Documentation & Presentation Reference Guide

This document provides a comprehensive technical overview and presentation slides guide for the **AI-Based Automated Optical Inspection (AOI) System**. It is designed to help developers, AI engineers, and presenters explain the project goals, system architecture, deep learning models, industrial UI design, and reporting engine.

---

## 1. Executive Summary & Problem Statement

In electronic assembly manufacturing (SMT/THT lines), verifying populated Printed Circuit Boards (PCBs) manually is slow, error-prone, and expensive. 

The **AI-Based Automated Optical Inspection (AOI) System** solves this by delivering an automated, dual-engine computer vision console that:
1. **Performs Real-Time Component Detection & Inventory**: Detects all physical components (`resistors`, `capacitors`, `ICs`, `connectors`, `LEDs`, `diodes`, `switches`, `transistors`, etc.), extracts their coordinates, calculates average confidence, and compiles an automated inventory ledger.
2. **Scans for Circuit & Solder Joint Defects**: Concurrently routes board images to specialized deep learning models (`DeepPCB`, `DsPCBSD+`, `HRIPCB`, `TDD-PCB`) to detect trace cracks, short circuits, open traces, mousebites, and solder fractures.
3. **Exports Certified Quality Reports**: Automatically generates multi-sheet Excel workbooks (`.xlsx`), CSV tabular ledgers, ReportLab PDF certificates, and JSON logs.

---

## 2. Dual-Engine Inspection Pipeline Architecture

```text
                            RAW PCB IMAGE
                                  │
                                  ▼
                   ┌──────────────┴──────────────┐
                   │  IMAGE ACQUISITION PANEL    │
                   │ (Validation & Memory Load)  │
                   └──────────────┬──────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
  ┌───────────────────────────────┐ ┌───────────────────────────────┐
  │  COMPONENT DETECTION ENGINE   │ │    CIRCUIT DEFECT ENGINE      │
  │  Model: Component_best.pt     │ │    Auto-Bound Profile Model   │
  │  Classes: 22 Component Types  │ │  (DeepPCB, DsPCBSD+, HRIPCB..)│
  └───────────────┬───────────────┘ └───────────────┬───────────────┘
                  │                                 │
                  ▼                                 ▼
   Raw Component Detections (45)        Defect Anomalies & Cracks
                  │                                 │
                  ├─────────────────────────────────┘
                  ▼
   ┌───────────────────────────────┐
   │ INDUSTRIAL AOI CONTROL CONSOLE│
   │ - KPI Cards (Total/Types/Conf)│
   │ - 2-Column Summary/Visuals    │
   │ - Sorted BBox Details Table   │
   │ - Circuit Defect Register     │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │ POLYMORPHIC REPORT EXPORTER   │
   │ (Multi-Sheet Excel, CSV, PDF) │
   └───────────────────────────────┘
```

---

## 3. Deep Learning Models & AI Specs

The system coordinates **5 specialized YOLO models**:

### A. Common Component Detector Model
- **Model Architecture**: YOLOv11 Deep Learning Object Detector
- **Weights Path**: `models/trained/Component/Component_best.pt`
- **Supported Classes (22)**: `battery`, `button`, `buzzer`, `capacitor`, `clock`, `connector`, `diode`, `display`, `fuse`, `heatsink`, `ic`, `inductor`, `led`, `pads`, `pins`, `potentiometer`, `relay`, `resistor`, `switch`, `transducer`, `transformer`, `transistor`.
- **Default Confidence Cutoff**: `0.25` (Optimized to detect 100% of physical components including small SMD parts).

### B. Profile-Specific Defect Detector Models
Selecting a PCB profile auto-binds its dedicated defect scanning model:
1. **Arduino Uno Profile (`arduino_uno`)** $\rightarrow$ **DeepPCB Model** (`models/trained/DeepPCB/DeepPCB.pt`)
   - *Scans*: `open`, `short`, `mousebite`, `spur`, `copper`, `pin_hole`.
2. **ESP32 DevKit Profile (`esp32_devkit`)** $\rightarrow$ **DsPCBSD+ Model** (`models/trained/DsPCBSD+/DsPCBSD+.pt`)
   - *Scans*: `SH`, `SP`, `SC`, `OP`, `MB`, `HB`, `CS`, `CFO`, `BMFO`.
3. **STM32 Blue Pill Profile (`stm32_blue_pill`)** $\rightarrow$ **HRIPCB Model** (`models/trained/HRIPCB/HRIPCB.pt`)
   - *Scans*: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`.
4. **Generic PCB Profile (`generic_pcb`)** $\rightarrow$ **TDD-PCB Model** (`models/trained/TDD-PCB/PDD-PCB-best.pt`)
   - *Scans*: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`.

---

## 4. Industrial AOI Dashboard Console Specifications

The dashboard console (`src/app/main.py`) provides:

1. **Top KPI Metrics Panel**:
   - **Total Components Detected**: Exact count of component bounding boxes (e.g. `45`).
   - **Unique Component Types**: Count of distinct classes present (e.g. `9`).
   - **Average Confidence**: Mean score across all component predictions (e.g. `74.2%`).
2. **Two-Column Inventory Panel**:
   - **Component Type Summary (Left)**: Grouped count per component class + `TOTAL` summary row + automated consistency verification ($\sum \text{Counts} == \text{Total}$).
   - **Component Visualization (Right)**: Large annotated PCB image displaying bright green bounding boxes (`#00FF66`) labeled with `<class_name> <confidence>`.
3. **Sorted Component Details Table**:
   - Lists `#`, `Component`, `Class ID`, `Confidence %`, `X1`, `Y1`, `X2`, `Y2`, and `Center (x%, y%)`.
   - Automatically sorted by confidence descending.
4. **Circuit Defect Register Tab**:
   - Independent registry tracking solder joint cracks and trace faults.

---

## 5. Reporting Engine & Multi-Sheet Excel Structure

The `ReportExporterFactory` generates print-ready certificates and machine-readable data:

- **Excel Workbook (`.xlsx`)**:
  - `Sheet 1 (Summary)`: PCB Profile Name, Image File Name, Timestamp, Model Path, Total Detections, Unique Types, Average Confidence, Confidence/IoU Thresholds, Hardware Device.
  - `Sheet 2 (Component Summary)`: Component Class vs Count table.
  - `Sheet 3 (Detection Details)`: Complete inventory ledger containing Bounding Boxes (`X1, Y1, X2, Y2`), Confidences, and Center % coordinates.
- **CSV Ledger (`.csv`)**: Flat CSV table for ERP and warehouse inventory ingestion.
- **PDF Certificate (`.pdf`)**: Styled ReportLab document containing summary tables and timestamps.
- **JSON Payload (`.json`)**: Database-ready structured payload.

---

## 6. Slide-by-Slide Presentation Outline Guide

Use this structure for presenting the project to stakeholders, faculty, or clients:

- **Slide 1: Title & Overview**
  - *Title*: Automated Optical Inspection (AOI) System for PCB Assembly Verification
  - *Subtitle*: Deep Learning-Powered Component Inventory & Circuit Defect Scanner
  - *Presenter Names & Roles*

- **Slide 2: Industry Problem & Motivation**
  - Manual inspection of populated PCBs is slow and prone to human error.
  - SMD components are microscopic; trace cracks cannot be seen with the naked eye.
  - Need for automated real-time detection, component counting, and defect classification.

- **Slide 3: System Solution & Dual-Pipeline Architecture**
  - Show Pipeline Architecture Diagram (Section 2).
  - Pipeline 1: YOLO Component Detection & Inventory.
  - Pipeline 2: Profile-Bound Defect Inspection (DeepPCB, DsPCBSD+, HRIPCB, TDD-PCB).

- **Slide 4: Deep Learning Models & AI Specs**
  - Highlight the 5 YOLO models (1 Component + 4 Defect Models).
  - Show the 22 component classes and defect classes.
  - Mention CUDA GPU acceleration and `ModelManager` caching layer.

- **Slide 5: Industrial Dashboard Console & UX**
  - Show Dashboard Highlights / Screenshot.
  - Explain KPI Cards: Total Components, Unique Types, Average Confidence.
  - Explain Two-Column Layout (Summary Table + Large Bounding Box Overlays).

- **Slide 6: Multi-Format Reporting & Export Factory**
  - Multi-sheet Excel workbook (`.xlsx`) structure.
  - PDF certificates, CSV ledgers, JSON database logs.

- **Slide 7: Verification & Test Results**
  - Show 100% Pass Rate across all 4 test suites (`validate_run.py`, `test_thresholds.py`, `test_workflow_scenarios.py`, `diagnose_full_pipeline.py`).
  - Demonstrate real reference image test results (`45/45` component detections on Arduino Uno).

- **Slide 8: Conclusion & Future Scope**
  - Production-ready AOI infrastructure deployed and verified.
  - Future Roadmap: OpenCV homography pre-processor & template-matching verification.

