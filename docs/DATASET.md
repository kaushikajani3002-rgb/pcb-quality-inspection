# PCB Dataset Configuration & Structure

To keep this repository lightweight, the raw PCB defect and component datasets are intentionally **excluded** from commits. The complete dataset footprint is approximately **8–9 GB**.

If you wish to retrain the models or run local validation, you must acquire or prepare a dataset in the standard YOLO format and structure it as described below.

---

## 1. Expected Directory Layout

The workspace should be structured with the datasets placed at the same level as the `PCB_AOI` project root:

```text
workspace/
├── PCB_AOI/                        # This repository (Source Code)
│   ├── app/
│   ├── config/
│   ├── inspection/
│   ├── templates/
│   ├── utils/
│   └── ...
└── datasets/                       # Datasets Directory (Excluded from git)
    ├── components/                 # Component detection dataset
    │   ├── data.yaml
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── valid/
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/
    │       ├── images/
    │       └── labels/
    └── defects/                    # Defect detection/segmentation dataset
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        └── valid/
            ├── images/
            └── labels/
```

---

## 2. YOLO Dataset Format

Each image folder must have a corresponding `labels` folder containing a text file for each image (with the exact same name, replacing the extension with `.txt`).

For **Object Detection** (e.g. Component Counting), each line in a label `.txt` file represents a bounding box in normalized coordinates:
```text
<class_id> <x_center_normalized> <y_center_normalized> <width_normalized> <height_normalized>
```
*Note: Values must be floats between `0.0` and `1.0` relative to the image size.*

For **Segmentation** (e.g. Solder Joint Segmentation), each line represents a polygon mask:
```text
<class_id> <x1_normalized> <y1_normalized> <x2_normalized> <y2_normalized> ... <xn_normalized> <yn_normalized>
```

---

## 3. Dataset Configuration (`data.yaml`)

The `data.yaml` configures the paths to dataset splits and names the classes.

Example `data.yaml` for Component Counting:
```yaml
path: ../datasets/components  # Dataset root path
train: train/images          # Relative path to training images
val: valid/images            # Relative path to validation images
test: test/images            # Relative path to test images (optional)

nc: 22                       # Number of classes
names:
  0: battery
  1: button
  2: buzzer
  3: capacitor
  4: clock
  5: connector
  6: diode
  7: display
  8: fuse
  9: heatsink
  10: ic
  11: inductor
  12: led
  13: pads
  14: pins
  15: potentiometer
  16: relay
  17: resistor
  18: switch
  19: transducer
  20: transformer
  21: transistor
```

---

## 4. How to Use Your Own Dataset

1. Prepare your images and labels in the YOLO directory structure shown in Section 1.
2. Annotate your images using tools like [CVAT](https://github.com/opencv/cvat) or [LabelImg](https://github.com/HumanSignal/labelImg) and export to **YOLO format**.
3. Create your custom `data.yaml` mapping your class indices to their text labels.
4. Modify the `--data` flag when executing `train_component_yolo.py`:
   ```bash
   python train_component_yolo.py --data /path/to/your/custom_data.yaml --epochs 50
   ```
5. Ensure the component type names in your custom `data.yaml` match the `"type"` fields declared inside your PCB layout profiles (e.g. `templates/arduino_uno.json`), or update the `CLASS_NAME_MAP` dictionary inside `detection_engine.py` to translate them at runtime.
