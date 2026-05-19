"""
src/predict.py
Real-time emotion detection from webcam using trained CNN + Haar cascade.

Usage:
    python src/predict.py
    python src/predict.py --model-path models/best_model.keras --camera 0
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import tensorflow as tf

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Angry":    (0,   0,   255),
    "Disgust":  (0,   128, 0),
    "Fear":     (128, 0,   128),
    "Happy":    (0,   255, 255),
    "Sad":      (255, 0,   0),
    "Surprise": (0,   165, 255),
    "Neutral":  (200, 200, 200),
}
IMG_SIZE = 48

# Haar cascade bundled with OpenCV
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time emotion detection")
    p.add_argument("--model-path", default=os.path.join("models", "best_model.keras"))
    p.add_argument("--camera",     type=int, default=0, help="Camera device index")
    p.add_argument("--scale",      type=float, default=1.3, help="Haar scaleFactor")
    p.add_argument("--min-neighbors", type=int, default=5)
    p.add_argument("--confidence", type=float, default=0.4, help="Min softmax confidence to display")
    return p.parse_args()


def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """Resize, normalise, expand dims for model input."""
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    return face


def draw_overlay(frame, x, y, w, h, label, confidence, color):
    """Draw bounding box + emotion label on frame."""
    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    text = f"{label}: {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)

    # Label text
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    args = parse_args()

    # ── Load model ───────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found at {args.model_path}")
        print("[INFO]  Run 'python src/train.py' first.")
        sys.exit(1)

    print(f"[INFO] Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    # ── Load face detector ───────────────────────────────────
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print("[ERROR] Failed to load Haar cascade.")
        sys.exit(1)

    # ── Open camera ──────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        sys.exit(1)

    print("[INFO] Starting real-time detection. Press 'q' to quit.")

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Retrying ...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect faces ─────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.min_neighbors,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y: y + h, x: x + w]
            face_input = preprocess_face(face_roi)

            preds = model.predict(face_input, verbose=0)[0]
            idx   = int(np.argmax(preds))
            conf  = float(preds[idx])
            label = EMOTION_LABELS[idx]
            color = EMOTION_COLORS[label]

            if conf >= args.confidence:
                draw_overlay(frame, x, y, w, h, label, conf, color)

                # Small bar chart of all emotions
                bar_x = x + w + 10
                for i, (em, prob) in enumerate(zip(EMOTION_LABELS, preds)):
                    bar_len = int(prob * 100)
                    bar_y   = y + i * 18
                    if bar_x + bar_len < frame.shape[1]:
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 12),
                                      EMOTION_COLORS[em], -1)
                        cv2.putText(frame, f"{em[:3]} {prob:.0%}",
                                    (bar_x + bar_len + 3, bar_y + 11),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # ── FPS counter ──────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection  [q to quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
