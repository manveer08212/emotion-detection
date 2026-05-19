"""
utils/augmentation.py
Keras ImageDataGenerator wrapper for FER-2013 augmentation.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_generator(X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 64):
    """
    Return augmented training generator.

    Augmentations applied:
        - Horizontal flip
        - ±10° rotation
        - ±10% width/height shift
        - ±10% zoom
        - Brightness jitter [0.8, 1.2]
    """
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    datagen.fit(X_train)
    return datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)


def get_val_generator(X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 64):
    """Return validation generator (no augmentation)."""
    datagen = ImageDataGenerator()
    return datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
