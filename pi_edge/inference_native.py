import ncnn
import cv2
import numpy as np
import time
from pathlib import Path

# --- PURE NATIVE NCNN INFERENCE (NO ULTRALYTICS/TORCH) ---
# This is completely immune to "Illegal instruction" errors!

class TubeDetectorNative:
    def __init__(self, model_path_dir, labels):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4 
        
        model_path = Path(model_path_dir)
        self.net.load_param(str(model_path / "model.ncnn.param"))
        self.net.load_model(str(model_path / "model.ncnn.bin"))
        self.labels = labels

    def detect(self, original_frame):
        img_h, img_w = original_frame.shape[:2]
        
        # Calculate letterbox scaling to 640x640
        scale = min(640 / img_w, 640 / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        pad_w = (640 - new_w) // 2
        pad_h = (640 - new_h) // 2
        
        # Resize and pad using NCNN optimized functions
        mat = ncnn.Mat.from_pixels_resize(original_frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, new_w, new_h)
        mat_pad = ncnn.Mat()
        ncnn.copy_make_border(mat, mat_pad, pad_h, 640 - new_h - pad_h, pad_w, 640 - new_w - pad_w, ncnn.BorderType.BORDER_CONSTANT, 0.0)
        
        mat_pad.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])

        ex = self.net.create_extractor()
        ex.input("in0", mat_pad)
        
        # In YOLO NCNN export, output layer is usually "out0"
        ret, out0 = ex.extract("out0")
        
        out_np = np.array(out0)
        # NCNN output for YOLO is typically a matrix. (9 rows, 8400 columns for 5 classes)
        if len(out_np.shape) == 3: out_np = out_np[0]
        if out_np.shape[0] == 9: out_np = out_np.T # Transpose to shape: (8400, 9)
        
        boxes = []
        scores = []
        class_ids = []
        
        # Verify shape (1, 8400, 9) => (8400, 9) where 9 = 4 box + 5 classes
        if len(out_np.shape) == 2 and out_np.shape[1] >= 5:
            probs = out_np[:, 4:]
            cls_ids = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            
            mask = confs > 0.60  # Confidence threshold
            valid_boxes = out_np[mask, :4]
            valid_confs = confs[mask]
            valid_cls_ids = cls_ids[mask]
            
            for i in range(len(valid_boxes)):
                cx, cy, bw, bh = valid_boxes[i]
                
                # Unpad and Descale back to original image size
                cx = (cx - pad_w) / scale
                cy = (cy - pad_h) / scale
                bw = bw / scale
                bh = bh / scale
                
                # Convert from center coordinates to top-left coordinates
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                
                boxes.append([x, y, int(bw), int(bh)])
                scores.append(float(valid_confs[i]))
                class_ids.append(valid_cls_ids[i])
                
        # NMS (Non-Maximum Suppression) to remove duplicate bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.60, nms_threshold=0.45)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], class_ids[i], scores[i]))
                
        return results

if __name__ == "__main__":
    MODEL_PATH = "best_ncnn_model" 
    LABELS = ["tube", "scratch", "crack", "bend", "hole"]
    
    print("🚀 Initializing 100% PURE Native AI Engine on 64-bit Pi...")
    try:
        detector = TubeDetectorNative(MODEL_PATH, LABELS)
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            start = time.time()
            
            # Run pure native inference
            results = detector.detect(frame)
            
            fps = 1 / (time.time() - start)
            
            # Draw boxes manually (No Ultralytics needed!)
            for (x, y, w, h, cls_id, score) in results:
                label = f"{LABELS[cls_id]}: {score:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show FPS
            cv2.putText(frame, f"Pi-Native FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Smart Factory Vision", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Hint: Make sure you use 'pip install ncnn numpy opencv-python-headless'")
