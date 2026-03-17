import os
import subprocess
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def print_ram(stage):
    pid = os.getpid()
    out = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)])
    # On macOS, ps rss is in KB. 
    rss_mb = int(out.strip()) / 1024.0
    print(f"[{stage}] RAM Usage: {rss_mb:.2f} MB")

if __name__ == '__main__':
    print_ram("Init")
    
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "runs/detect/brand_experiment2/weights/best.pt"
    
    print("Loading model...")
    if model_path.exists():
        model = YOLO(model_path)
    else:
        model = YOLO("yolo11n.pt")
        
    print_ram("After loading model")
    
    print("Running dummy inference...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy_frame, verbose=False)
    
    print("Running second dummy inference...")
    model.predict(dummy_frame, verbose=False)
    print_ram("After inference")
