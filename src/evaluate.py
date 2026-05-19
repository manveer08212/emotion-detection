"""
src/evaluate.py
Evaluate saved model on test set. Outputs confusion matrix + report.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model-path models/best_model.keras
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf

from utils.preprocessing import load_processed
from utils.visualization import (
    plot_confusion_matrix,
    print_classification_report,
    plot_sample_predictions,
)

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Emotion CNN")
    p.add_argument("--model-path", default=os.path.join("models", "best_model.keras"))
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    print(f"[INFO] Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    print("[INFO] Loading test data ...")
    _, _, (X_test, y_test) = load_processed()

    # ── Predictions ──────────────────────────────────────────
    print("[INFO] Running predictions ...")
    y_prob = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ── Metrics ──────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=0)
    print(f"\n[RESULT] Test Loss={loss:.4f}  Test Accuracy={acc:.4f} ({acc*100:.2f}%)")

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png")
    plot_sample_predictions(X_test, y_true, y_pred, n=12, save_path="sample_predictions.png")

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
