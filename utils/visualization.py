"""
utils/visualization.py
Plotting helpers: training curves, confusion matrix, per-class accuracy.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def plot_training_history(history, save_path: str = "training_history.png"):
    """Plot accuracy & loss curves from Keras history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Saved training history → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    """Plot normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalised Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Saved confusion matrix → {save_path}")
    plt.show()


def print_classification_report(y_true, y_pred):
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))


def plot_sample_predictions(X, y_true, y_pred, n: int = 12, save_path: str = "sample_preds.png"):
    """Display grid of sample images with true vs predicted labels."""
    indices = np.random.choice(len(X), n, replace=False)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].squeeze(), cmap="gray")
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        axes[i].set_title(
            f"T:{EMOTION_LABELS[y_true[idx]]}\nP:{EMOTION_LABELS[y_pred[idx]]}",
            color=color, fontsize=8
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Saved sample predictions → {save_path}")
    plt.show()
