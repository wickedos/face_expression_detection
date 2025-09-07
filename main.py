# main.py
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os
import time
import sys

# -------- Configuration (edit if you trained with a different label order) --------
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
# ---------------------------------------------------------------------------------

# Load OpenCV cascade from cv2 installation (no local XML required)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Cascade loaded? ->", not face_cascade.empty())

# Load model from same directory as this script
model_path = os.path.join(os.path.dirname(__file__), "model.h5")
if not os.path.exists(model_path):
    print("ERROR: model.h5 not found at:", model_path)
    sys.exit(1)

classifier = load_model(model_path)
print("Loaded model from:", model_path)
print("Model summary (partial):")
try:
    classifier.summary()
except Exception:
    print("Could not print full summary.")

# Detect model expected input shape
# Typical TF Keras shape: (None, height, width, channels)
input_shape = getattr(classifier, "input_shape", None)
print("Detected model.input_shape:", input_shape)

# Default fallback assumptions:
target_h, target_w, target_c = 48, 48, 1

if input_shape and len(input_shape) >= 3:
    # handle shapes like (None, H, W, C) or (None, C, H, W) (rare for TF)
    if len(input_shape) == 4:
        _, h, w, c = input_shape
        # If any are None, keep fallback
        if all(isinstance(x, int) and x > 0 for x in (h, w, c)):
            target_h, target_w, target_c = h, w, c
    elif len(input_shape) == 3:
        # (H,W,C) without batch dim
        h, w, c = input_shape
        if all(isinstance(x, int) and x > 0 for x in (h, w, c)):
            target_h, target_w, target_c = h, w, c

print(f"Preprocessing target: {target_h}x{target_w}x{target_c}")

# Setup camera (try indices 0..3 until one opens)
def open_camera(preferred_index=0, max_index=3):
    for idx in range(preferred_index, max_index+1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print("Opened camera index:", idx)
            return cap
        else:
            cap.release()
    return None

cap = open_camera(0, 2)
if cap is None:
    print("ERROR: Could not open any camera (tried indices 0..2). Exiting.")
    sys.exit(1)

def preprocess_for_model(face_img):
    """
    face_img: numpy array in BGR colorspace (as from OpenCV), possibly grayscale if single channel.
    Returns: array shaped (1, H, W, C) suitable for classifier.predict
    """
    # If model expects color (3 channels) but face_img is gray, convert to BGR first:
    if len(face_img.shape) == 2 and target_c == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    # If face_img is BGR but model expects grayscale (1), convert:
    if len(face_img.shape) == 3 and target_c == 1:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Resize to model expected size
    if target_c == 1:
        resized = cv2.resize(face_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        arr = resized.astype('float32') / 255.0
        arr = img_to_array(arr)  # will produce (H,W,1)
    else:
        # target_c == 3
        resized = cv2.resize(face_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        arr = resized.astype('float32') / 255.0
        arr = img_to_array(arr)  # (H,W,3)

    # Ensure shape (1,H,W,C)
    arr = np.expand_dims(arr, axis=0)
    return arr

print("Starting main loop. Press 'q' in the window to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't read from camera. Exiting loop.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            # show frame and wait briefly
            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed.")
                break
            # check if user closed the window
            if cv2.getWindowProperty('Emotion Detector', cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break
            continue

        # select largest face
        x, y, w, h = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]

        # extract the face region (use original colored frame for color models)
        if target_c == 3:
            face_region = frame[y:y+h, x:x+w]            # BGR
            face_display = cv2.resize(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), (200,200))
        else:
            face_region = gray[y:y+h, x:x+w]             # grayscale
            face_display = cv2.resize(face_region, (200,200))

        # show face crop for debugging
        cv2.imshow('FaceCrop-debug', face_display)

        # Preprocess according to model input
        inp = preprocess_for_model(face_region)
        print("DEBUG: input to model shape:", inp.shape)

        # Predict
        preds = classifier.predict(inp)[0]
        print("DEBUG: raw preds:", preds)

        # Interpret predictions
        top_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else f"class_{top_idx}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        label_position = (x, y - 10 if y-10 > 10 else y + 20)
        cv2.putText(frame, f"{label} {confidence:.2f}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Save debug face (overwrite)
        try:
            if target_c == 3:
                # save grayscale version for quick viewing
                cv2.imwrite("debug_face.jpg", cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
            else:
                cv2.imwrite("debug_face.jpg", face_region)
        except Exception as e:
            print("Could not write debug_face.jpg:", e)

        # Show the annotated frame
        cv2.imshow('Emotion Detector', frame)

        # Quit conditions
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed.")
            break
        # If user closed window:
        if cv2.getWindowProperty('Emotion Detector', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user.")
            break

        # tiny throttle
        time.sleep(0.02)

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Clean exit.")
