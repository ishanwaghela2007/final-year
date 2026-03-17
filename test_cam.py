import cv2
import numpy as np
print("Attempting to open camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera 0")
else:
    print("Camera 0 opened successfully.")
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame!")
    else:
        print("Failed to read a frame.")
    cap.release()
print("Attempting to show a window...")
cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
cv2.imshow('Test Window', np.zeros((100, 100, 3), dtype=np.uint8))
print("Press any key in the window to close...")
cv2.waitKey(2000)
cv2.destroyAllWindows()
print("Done.")
