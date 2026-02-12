import cv2
import numpy as np
import subprocess
import os
import urllib.request

# ==========================================
# CONFIGURATION
# ==========================================
# 1. AI Settings (MobileNet)
CONFIDENCE_THRESHOLD = 0.5
SKIP_FRAMES = 5   # Run AI every 5th frame for speed
AI_CLASSES_TO_DETECT = ["boat"] # User requested ONLY boats

# 2. Color Settings (Orange Floaters)
# Orange is typically Hue 10-25 in OpenCV
LOWER_ORANGE = np.array([10, 100, 100])
UPPER_ORANGE = np.array([25, 255, 255])
MIN_FLOATER_AREA = 300

# 3. Model Files
FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel"
}

# Standard Labels for MobileNet
CLASSES = ["bg", "plane", "bike", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "moto", "person", "plant", "sheep", "sofa", "train", "tv"]
# Fixed Colors: Boat (Blue), Orange Floater (Orange)
COLOR_BOAT = (255, 0, 0)    # Blue in BGR
COLOR_ORANGE = (0, 165, 255) # Orange in BGR

def main():
    # --- Setup ---
    for f, url in FILES.items():
        if not os.path.exists(f):
            print(f"Downloading {f}...")
            urllib.request.urlretrieve(url, f)
            
    print("[INFO] Loading AI...")
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

    # --- Start Camera (rpicam-vid / MJPEG) ---
    print("[INFO] Starting Camera...")
    cmd = ["rpicam-vid", "-t", "0", "--inline", "--width", "640", "--height", "480", "--codec", "mjpeg", "--nopreview", "-o", "-"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    buffer = b""
    frame_count = 0
    
    # Cache for frame skipping
    last_ai_boxes = [] 
    last_floater_boxes = []

    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk: break
            buffer += chunk

            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = buffer[a:b+2]
                buffer = buffer[b+2:]
                
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue

                (h, w) = frame.shape[:2]
                frame_count += 1

                # =================================================
                # 1. DETECTION LOGIC (Runs every SKIP_FRAMES)
                # =================================================
                if frame_count % SKIP_FRAMES == 0:
                    last_ai_boxes = []
                    last_floater_boxes = []

                    # --- A. Detect BOATS (AI) ---
                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()

                    for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > CONFIDENCE_THRESHOLD:
                            idx = int(detections[0, 0, i, 1])
                            if CLASSES[idx] in AI_CLASSES_TO_DETECT:
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")
                                last_ai_boxes.append((startX, startY, endX, endY, "BOAT"))

                    # --- B. Detect ORANGE FLOATERS (Color) ---
                    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    
                    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
                    mask = cv2.erode(mask, None, iterations=2)
                    mask = cv2.dilate(mask, None, iterations=2)
                    
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        if cv2.contourArea(c) > MIN_FLOATER_AREA:
                            x, y, wb, hb = cv2.boundingRect(c)
                            # Shape Filter: Floaters shouldn't be super long thin lines
                            ar = float(wb) / hb
                            if 0.4 < ar < 2.0:
                                last_floater_boxes.append((x, y, x+wb, y+hb))

                # =================================================
                # 2. DRAWING (Runs EVERY frame)
                # =================================================
                # Draw Boats
                for (startX, startY, endX, endY, label) in last_ai_boxes:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR_BOAT, 2)
                    cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOAT, 2)

                # Draw Orange Floaters
                for (x, y, x2, y2) in last_floater_boxes:
                    cv2.rectangle(frame, (x, y), (x2, y2), COLOR_ORANGE, 2)
                    cv2.putText(frame, "FLOATER", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ORANGE, 2)

                cv2.imshow("Marine: Boats & Orange Floaters", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
