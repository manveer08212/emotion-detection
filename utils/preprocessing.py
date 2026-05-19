"""
utils/preprocessing.py
Load & preprocess FER-2013 dataset from CSV → numpy arrays.

Usage:
    python utils/preprocessing.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ── Constants ────────────────────────────────────────────────
RAW_CSV   = os.path.join("data", "raw", "fer2013.csv")
PROC_DIR  = os.path.join("data", "processed")
IMG_SIZE  = 48
NUM_CLASSES = 7


def load_fer2013(csv_path: str = RAW_CSV):
    """
    Parse FER-2013 CSV into pixel arrays + labels.

    Returns:
        X (N, 48, 48, 1) float32 in [0, 1]
        y (N,) int labels
        usage list of strings
    """
    print(f"[INFO] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Pixel string → numpy array
    pixels = df["pixels"].apply(
        lambda s: np.fromstring(s, dtype=np.uint8, sep=" ").reshape(IMG_SIZE, IMG_SIZE)
    )
    X = np.stack(pixels.values)[..., np.newaxis].astype(np.float32)
    y = df["emotion"].values.astype(np.int32)

    print(f"[INFO] Loaded {len(X)} samples. Class dist: {np.bincount(y)}")
    return X, y, df["Usage"].values


def normalize(X: np.ndarray) -> np.ndarray:
    """Scale pixels to [0, 1]."""
    return X / 255.0


def per_image_normalize(X: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per image (stronger preprocessing)."""
    mean = X.mean(axis=(1, 2, 3), keepdims=True)
    std  = X.std(axis=(1, 2, 3), keepdims=True) + 1e-7
    return (X - mean) / std


def to_categorical(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return tf.keras.utils.to_categorical(y, num_classes)


def split_by_usage(X, y, usage):
    """Split using FER-2013 built-in Usage column."""
    train_mask = usage == "Training"
    val_mask   = usage == "PublicTest"
    test_mask  = usage == "PrivateTest"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_processed(X_train, y_train, X_val, y_val, X_test, y_test, out_dir: str = PROC_DIR):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)
    print(f"[INFO] Saved processed arrays to {out_dir}/")


def load_processed(proc_dir: str = PROC_DIR):
    """Load pre-saved numpy arrays."""
    X_train = np.load(os.path.join(proc_dir, "X_train.npy"))
    y_train = np.load(os.path.join(proc_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(proc_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(proc_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(proc_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(proc_dir, "y_test.npy"))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    X, y, usage = load_fer2013()
    X = normalize(X)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_usage(X, y, usage)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    y_train = to_categorical(y_train)
    y_val   = to_categorical(y_val)
    y_test  = to_categorical(y_test)

    save_processed(X_train, y_train, X_val, y_val, X_test, y_test)
    print("[INFO] Preprocessing complete.")
