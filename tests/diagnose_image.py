import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    model_path = r"D:\PCB\PCB_AOI\models\trained\Component\All_cercit_finetuned_best.pt"
    image_path = r"C:\Users\kaushik ajani\.gemini\antigravity\brain\61fa869f-a86d-4b96-a564-b120d3c14c97\.user_uploaded\media__1785425441719.png"
    
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        # Try the other uploaded image
        image_path = r"C:\Users\kaushik ajani\.gemini\antigravity\brain\61fa869f-a86d-4b96-a564-b120d3c14c97\.user_uploaded\media_1786721784073.png"
        if not os.path.exists(image_path):
            print(f"Second image path does not exist: {image_path}")
            return
            
    print(f"Running inference on: {image_path}")
    model = YOLO(model_path)
    
    # Run at conf=0.1
    results = model.predict(source=image_path, conf=0.1, imgsz=640)
    boxes = results[0].boxes
    print(f"Total detections at conf=0.1: {len(boxes)}")
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names.get(cls_id, f"Class {cls_id}")
        xyxy = box.xyxy[0].tolist()
        print(f"  Det {i+1}: Class='{class_name}', Conf={conf:.4f}, Box={xyxy}")

if __name__ == "__main__":
    main()
