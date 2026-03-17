import cv2
import os
import time

camera_index = 0
save_dir = "dataset_capture"

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(camera_index)

capture_interval = 8  # seconds
img_count = 0

print("Starting tube image capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    cv2.imshow("Tube Capture", frame)

    timestamp = int(time.time())
    filename = f"{save_dir}/tube_{timestamp}_{img_count}.jpg"

    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

    img_count += 1

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

    time.sleep(capture_interval)

cap.release()
cv2.destroyAllWindows()
