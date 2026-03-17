"""
Real-time Tube Detection with Tracking
======================================
- Prefers ONNX model (then NCNN, TFLite, PyTorch)
- 5-second cooldown prevents duplicate counts
- Unique IDs: TUBE_001, TUBE_002, ...
- Only tube classes, confidence > 0.7
- ByteTrack tracking (same tube = one count)
- Optimized for Raspberry Pi
"""

import cv2
import argparse
import sqlite3
import time
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from ultralytics import YOLO


def _normalize_name(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "").replace("(", "").replace(")", "")


# ─────────────────────────────────────────────────────────────────────────────
# Threaded Video Capture (Raspberry Pi)
# ─────────────────────────────────────────────────────────────────────────────
class VideoCaptureAsync:
    def __init__(self, source):
        if os.name == "nt" and isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.grabbed, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
def setup_db(conn: sqlite3.Connection):
    """Create tube_detections table: tube_id, timestamp, brand_name, confidence, track_id."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tube_detections (
            tube_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            brand_name TEXT,
            confidence REAL,
            track_id INTEGER
        )
    """)
    conn.commit()


def get_next_tube_id(conn: sqlite3.Connection) -> str:
    """TUBE_001, TUBE_002, ..."""
    cursor = conn.cursor()
    cursor.execute("SELECT tube_id FROM tube_detections ORDER BY tube_id DESC LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        return "TUBE_001"
    try:
        num = int(str(row[0]).replace("TUBE_", ""))
        return f"TUBE_{num + 1:03d}"
    except (ValueError, AttributeError):
        return "TUBE_001"


def log_tube(conn: sqlite3.Connection, tube_id: str, brand_name: str, confidence: float, track_id: int):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO tube_detections (tube_id, timestamp, brand_name, confidence, track_id)
        VALUES (?, ?, ?, ?, ?)
    """, (tube_id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), brand_name, round(confidence, 2), track_id))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Load valid tube classes (only tubes, ignore person/background/noise)
# ─────────────────────────────────────────────────────────────────────────────
def load_valid_classes(base_dir: Path) -> set:
    classes_file = base_dir / "data" / "dataset(tubes)" / "classes.txt"
    if not classes_file.exists():
        return {_normalize_name(c) for c in ["valbet", "araldite", "beutiful-n", "dk gel", "silverkant", "cani-maks"]}
    with open(classes_file, "r", encoding="utf-8") as f:
        return {_normalize_name(line.strip()) for line in f if line.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# Main System
# ─────────────────────────────────────────────────────────────────────────────
def run_system(
    source_input,
    imgsz=320,
    min_conf=0.7,
    frame_skip=2,
    min_box_area=3000,
    cooldown_sec=5.0,
    headless=False,
    run_name="balanced_run",
):
    base_dir = Path(__file__).resolve().parent.parent
    weights_dir = base_dir / "runs" / "detect" / run_name / "weights"

    # Prefer ONNX (faster on Pi), then NCNN, TFLite, PyTorch
    onnx_path = weights_dir / "best.onnx"
    ncnn_path = weights_dir / "best_ncnn_model"
    tflite_path = weights_dir / "best_saved_model" / "best_float32.tflite"
    pt_path = weights_dir / "best.pt"

    if onnx_path.exists():
        print("🚀 Loading ONNX model")
        model = YOLO(str(onnx_path), task="detect")
    elif ncnn_path.exists():
        print("🚀 Loading NCNN model")
        model = YOLO(str(ncnn_path), task="detect")
    elif tflite_path.exists():
        print("🚀 Loading TFLite model")
        model = YOLO(str(tflite_path), task="detect")
    elif pt_path.exists():
        print("✅ Loading PyTorch model")
        model = YOLO(str(pt_path))
    else:
        raise RuntimeError(
            f"No model in runs/detect/{run_name}/weights. "
            f"Export ONNX: model.export(format='onnx') or train with --name {run_name}"
        )

    normalized_valid = load_valid_classes(base_dir)
    alias_map = {
        "valbet": "valbet", "val-bet": "valbet", "val bet": "valbet",
        "silverkant": "silverkant", "silver-kant": "silverkant", "skkant": "silverkant",
        "dk_gel": "dk gel", "dk gel": "dk gel",
        "cani-maks": "cani-maks", "cani maks": "cani-maks",
        "beutiful-n": "beutiful-n", "beautiful-n": "beutiful-n",
        "araldite": "araldite", "halobet": "halobet",
    }

    db_path = base_dir / "data" / "tube_detections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    setup_db(conn)

    # Cooldown & tracking
    last_detection_time = 0.0
    logged_track_ids = set()
    tube_count = 0

    source_int = int(source_input) if str(source_input).isdigit() else source_input
    cap = VideoCaptureAsync(source_int) if isinstance(source_int, int) else cv2.VideoCapture(source_input)

    if not cap.isOpened():
        print("Camera not opened. Try --source 0 or 1")
        conn.close()
        return

    print("=" * 50)
    print(f"Tube Detection | Model: {run_name} | Cooldown: {cooldown_sec}s")
    print("Press Q to quit")
    print("=" * 50)

    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    display_fps = 0.0

    if not headless:
        cv2.namedWindow("ML Brand Detector", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            if not headless:
                cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("ML Brand Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        with torch.no_grad():
            results = model.track(
                frame,
                conf=min_conf,
                imgsz=imgsz,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

        best_detection = None
        best_conf = 0.0

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = (
                results[0].boxes.id.int().cpu().numpy()
                if results[0].boxes.id is not None
                else np.full(len(boxes), -1, dtype=int)
            )

            for box, cls_id, conf, track_id in zip(boxes, clss, confs, track_ids):
                if conf < min_conf:
                    continue
                raw_name = model.names.get(int(cls_id), "")
                norm = _normalize_name(raw_name)
                canonical = alias_map.get(norm, raw_name.lower())
                if _normalize_name(canonical) not in normalized_valid:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                if track_id >= 0 and track_id in logged_track_ids:
                    continue
                if conf > best_conf:
                    best_conf = float(conf)
                    best_detection = (canonical, float(conf), (int(x1), int(y1), int(x2), int(y2)), int(track_id) if track_id >= 0 else -1)

        now = time.time()
        cooldown_elapsed = (now - last_detection_time) >= cooldown_sec

        if best_detection and cooldown_elapsed:
            brand_name, conf, (x1, y1, x2, y2), track_id = best_detection
            tube_id = get_next_tube_id(conn)
            log_tube(conn, tube_id, brand_name, conf, track_id)
            if track_id >= 0:
                logged_track_ids.add(track_id)
            last_detection_time = now
            tube_count += 1
            print(f"[{tube_id}] {brand_name} | conf={conf:.2f} | total={tube_count}")
            if not headless:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{tube_id} {brand_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif best_detection and not headless:
            _, _, (x1, y1, x2, y2), _ = best_detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, "cooldown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(logged_track_ids) > 20:
            logged_track_ids.clear()

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            display_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tubes: {tube_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if cooldown_elapsed:
            cv2.putText(frame, "READY", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            rem = cooldown_sec - (now - last_detection_time)
            cv2.putText(frame, f"Cooldown: {rem:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if not headless:
            cv2.imshow("ML Brand Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    conn.close()
    print(f"\nDone. Total tubes: {tube_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="1", help="Camera index")
    parser.add_argument("--run", type=str, default="balanced_run", help="Model run (runs/detect/<run>/weights)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between detections")
    parser.add_argument("--min-conf", type=float, default=0.70, help="Min confidence")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size (320=fast)")
    parser.add_argument("--min-area", type=int, default=3000, help="Min box area")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    run_system(
        source_input=args.source,
        imgsz=args.imgsz,
        min_conf=args.min_conf,
        frame_skip=args.skip,
        min_box_area=args.min_area,
        cooldown_sec=args.cooldown,
        headless=args.headless,
        run_name=args.run,
    )
