import cv2
import os
from ultralytics import YOLO


def run_detection(camera_index: int = 1):
    # -------- Load model from best.pt next to this script --------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best.pt")  # copy your trained best.pt here
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Place your trained best.pt next to this script."
        )

    model = YOLO(model_path)

    # -------- Tube-only classes set (filter by name, not index) --------
    # These are canonical names from your dataset; we'll map model's class names to these.
    TUBE_CLASSES = {
        "araldite",
        "beutiful-n",
        "cani-maks",
        "dk_gel",
        "halobet",
        "silverkant",
        "valbet",
    }

    # Map common label variants to canonical tube names (if your model uses slightly different labels)
    alias_map = {
        "valbet": "valbet",
        "val-bet": "valbet",
        "val bet": "valbet",
        "silver-kant": "silverkant",
        "silverkant": "silverkant",
        "skkant": "silverkant",
        "dk gel": "dk_gel",
        "dk_gel": "dk_gel",
        "cani-maks": "cani-maks",
        "cani maks": "cani-maks",
        "beautiful-n": "beutiful-n",
        "beutiful-n": "beutiful-n",
    }

    def norm_name(s: str) -> str:
        return (
            s.strip().lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

    normalized_tubes = {norm_name(c) for c in TUBE_CLASSES}

    # High confidence
    CONFIDENCE_THRESHOLD = 0.90

    # TOTAL COUNTS (per class, for this run)
    total_counts = {cls: 0 for cls in sorted(TUBE_CLASSES)}

    # -------- Open external webcam (Windows: use DirectShow backend) --------
    is_windows = os.name == "nt"
    if is_windows:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(
            f"Camera at index {camera_index} not detected.\n"
            f"- On Windows, external USB is usually 1 (with DirectShow).\n"
            f"- Try other indices (0, 1, 2) or check USB/camera privacy settings."
        )
        return

    print("Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Skip low-confidence detections
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Get model's class name, normalize and map to canonical tube name
                raw_name = str(model.names.get(cls_id, "")).strip()
                norm = norm_name(raw_name)
                if norm not in normalized_tubes:
                    # Not one of our tube classes -> ignore (no people/other objects)
                    continue

                canonical = alias_map.get(raw_name.lower(), raw_name.lower())
                if canonical not in total_counts:
                    # If model name slightly different, fall back to a generic bucket
                    canonical = list(total_counts.keys())[0]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Increase TOTAL count for this class
                total_counts[canonical] += 1

        # -------- Dashboard overlay --------
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 10 + 30 + 25 * len(CLASSES)), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(
            frame,
            "DETECTION COUNTS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        y = 70
        for cls in sorted(total_counts.keys()):
            text = f"{cls}: {total_counts[cls]}"
            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )
            y += 25

        cv2.imshow("Tube Detection", frame)

        # ESC to exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Default to external camera index 1; change here if needed
    run_detection(camera_index=1)