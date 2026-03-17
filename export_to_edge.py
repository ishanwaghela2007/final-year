from ultralytics import YOLO
import sys
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "runs/detect/brand_experiment2/weights/best.pt"
    
    if not model_path.exists():
        print(f"Error: Could not find model at {model_path}.")
        print("Falling back to standard yolo11n.pt for demonstration...")
        model_path = "yolo11n.pt"
        
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("\n--- Exporting to TFLite ---")
    # Exports to TFLite (Float32 format). Extremely compatible with edge CPUs.
    try:
        model.export(format="tflite")
    except Exception as e:
        print(f"⚠️ TFLite export failed (this is common if TensorFlow isn't installed). Error: {e}")
        print("Skipping TFLite export. NCNN is usually better for Raspberry Pi anyway!")
    
    print("\n--- Exporting to NCNN ---")
    # NCNN is heavily optimized specifically for mobile/edge ARM CPUs like on the Raspberry Pi.
    model.export(format="ncnn")
    
    print("\n✅ Export complete! The `track_and_id.py` script has been updated to automatically detect and load these lightweight models instead of the heavy `.pt` model.")

if __name__ == "__main__":
    main()
