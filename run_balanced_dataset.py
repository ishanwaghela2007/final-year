"""
Run inference with the balanced_dataset trained model.
Loads best.pt from runs/detect/balanced_dataset/weights/
"""
import cv2
import os
from ultralytics import YOLO


def run_detection(camera_index: int = 1, run_name: str = "balanced_dataset"):
    # -------- Load model from runs/detect/<run_name>/weights/best.pt --------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "runs", "detect", run_name, "weights", "best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}\n"
            f"Train with: python src/train.py --name {run_name}\n"
            f"Or check that runs/detect/{run_name}/weights/best.pt exists."
        )

    model = YOLO(model_path)

    # -------- Tube classes (from model - we'll accept whatever the model predicts) --------
    # Build TUBE_CLASSES from model.names for flexibility
    TUBE_CLASSES = set(model.names.values()) if hasattr(model, 'names') and model.names else set()

    alias_map = {
        "valbet": "valbet", "val-bet": "valbet", "val bet": "valbet",
        "silver-kant": "silverkant", "silverkant": "silverkant", "skkant": "silverkant",
        "dk gel": "dk_gel", "dk_gel": "dk_gel",
        "cani-maks": "cani-maks", "cani maks": "cani-maks",
        "beautiful-n": "beutiful-n", "beutiful-n": "beutiful-n",
    }

    def norm_name(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    normalized_tubes = {norm_name(c) for c in TUBE_CLASSES} if TUBE_CLASSES else set()

    CONFIDENCE_THRESHOLD = 0.90
    total_counts = {cls: 0 for cls in sorted(TUBE_CLASSES)} if TUBE_CLASSES else {}

    # -------- Open camera --------
    is_windows = os.name == "nt"
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if is_windows else camera_index)

    if not cap.isOpened():
        print(
            f"Camera at index {camera_index} not detected.\n"
            f"On Windows, external USB is usually 1. Try 0, 1, 2 or check USB/camera settings."
        )
        return

    print(f"Running balanced_dataset model | Camera {camera_index} | Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                raw_name = str(model.names.get(cls_id, "")).strip()
                norm = norm_name(raw_name)
                if normalized_tubes and norm not in normalized_tubes:
                    continue

                canonical = alias_map.get(raw_name.lower(), raw_name.lower())
                if canonical not in total_counts:
                    total_counts[canonical] = 0

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{canonical} {conf:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                total_counts[canonical] += 1

        # -------- Dashboard overlay --------
        classes_list = sorted(total_counts.keys())
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 10 + 30 + 25 * len(classes_list)), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "DETECTION COUNTS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y = 70
        for cls in classes_list:
            cv2.putText(frame, f"{cls}: {total_counts[cls]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y += 25

        cv2.imshow("Tube Detection (balanced_dataset)", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--run", default="balanced_dataset", help="Run name under runs/detect/ (default: balanced_dataset)")
    args = parser.parse_args()
    run_detection(camera_index=args.camera, run_name=args.run)
