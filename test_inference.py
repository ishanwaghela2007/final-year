from ultralytics import YOLO
import cv2
import glob
from pathlib import Path

# Provide a sample image from the dataset
images = glob.glob("/Volumes/Untitled 2/desktop/final year project/ml/main/data/dataset(tubes)/images/*.jpg")
if not images:
    print("No images found in dataset!")
    exit(1)

test_img = images[0]
print(f"Testing on: {test_img}")

# Load model
model_path = "/Volumes/Untitled 2/desktop/final year project/ml/main/runs/detect/brand_experiment/weights/best.pt"
model = YOLO(model_path)

# Run inference
results = model.predict(test_img, conf=0.1)

print("\n--- RESULTS ---")
boxes = results[0].boxes
if len(boxes) == 0:
    print("NO BOXES DETECTED!")
else:
    print(f"Detected {len(boxes)} boxes.")
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        name = model.names[cls_id]
        print(f"- Class: {name} (ID: {cls_id}) | Confidence: {conf:.4f}")
