"""
src/train.py
Train CNN on FER-2013 dataset.

Usage:
    python src/train.py
    python src/train.py --epochs 80 --batch-size 32 --lr 5e-4
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from models.emotion_cnn import build_model, compile_model
from utils.preprocessing import load_processed
from utils.augmentation import get_train_generator, get_val_generator
from utils.visualization import plot_training_history


# ── CLI Args ─────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Emotion CNN")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--model-path",  type=str,   default=os.path.join("models", "best_model.keras"))
    parser.add_argument("--no-augment",  action="store_true", help="Disable data augmentation")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── GPU config ───────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using {len(gpus)} GPU(s)")
    else:
        print("[INFO] No GPU found — running on CPU")

    # ── Load data ────────────────────────────────────────────
    print("[INFO] Loading processed data ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_processed()
    print(f"[INFO] Train={X_train.shape[0]}  Val={X_val.shape[0]}  Test={X_test.shape[0]}")

    # ── Build model ──────────────────────────────────────────
    model = build_model()
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    # ── Callbacks ────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            args.model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(log_dir="logs/"),
    ]

    # ── Train ────────────────────────────────────────────────
    steps_per_epoch = len(X_train) // args.batch_size

    if args.no_augment:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
        )
    else:
        train_gen = get_train_generator(X_train, y_train, args.batch_size)
        val_gen   = get_val_generator(X_val, y_val, args.batch_size)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=callbacks,
        )

    # ── Evaluate on test set ─────────────────────────────────
    print("\n[INFO] Evaluating on test set ...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test Loss={test_loss:.4f}  Test Accuracy={test_acc:.4f}")

    # ── Plot curves ──────────────────────────────────────────
    plot_training_history(history, save_path="training_history.png")

    print(f"[INFO] Best model saved to {args.model_path}")


if __name__ == "__main__":
    main()
