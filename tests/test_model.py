"""
tests/test_model.py
Unit tests for model architecture and preprocessing utilities.

Run:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest


# ── Model tests ──────────────────────────────────────────────

class TestEmotionCNN:
    def test_build_model_output_shape(self):
        from models.emotion_cnn import build_model, NUM_CLASSES
        model = build_model()
        assert model.output_shape == (None, NUM_CLASSES)

    def test_build_model_input_shape(self):
        from models.emotion_cnn import build_model, IMG_SIZE
        model = build_model()
        assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 1)

    def test_compile_model(self):
        from models.emotion_cnn import build_model, compile_model
        model = build_model()
        model = compile_model(model)
        assert model.optimizer is not None

    def test_forward_pass(self):
        from models.emotion_cnn import build_model, compile_model, NUM_CLASSES, IMG_SIZE
        model = build_model()
        model = compile_model(model)
        dummy = np.random.rand(4, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
        preds = model.predict(dummy, verbose=0)
        assert preds.shape == (4, NUM_CLASSES)
        # Softmax outputs should sum to ~1
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)

    def test_custom_num_classes(self):
        from models.emotion_cnn import build_model
        model = build_model(num_classes=3)
        assert model.output_shape == (None, 3)


# ── Preprocessing tests ──────────────────────────────────────

class TestPreprocessing:
    def test_normalize(self):
        from utils.preprocessing import normalize
        X = np.array([[[128.0]]])
        result = normalize(X)
        assert result[0, 0, 0] == pytest.approx(128.0 / 255.0, abs=1e-6)

    def test_normalize_range(self):
        from utils.preprocessing import normalize
        X = np.random.randint(0, 256, (10, 48, 48, 1)).astype(np.float32)
        result = normalize(X)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_to_categorical(self):
        from utils.preprocessing import to_categorical
        y = np.array([0, 1, 6])
        result = to_categorical(y, num_classes=7)
        assert result.shape == (3, 7)
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0
        assert result[2, 6] == 1.0

    def test_per_image_normalize(self):
        from utils.preprocessing import per_image_normalize
        X = np.random.rand(5, 48, 48, 1).astype(np.float32)
        result = per_image_normalize(X)
        # Mean ≈ 0 per image
        means = result.mean(axis=(1, 2, 3))
        np.testing.assert_allclose(means, np.zeros(5), atol=1e-5)


# ── Augmentation tests ───────────────────────────────────────

class TestAugmentation:
    def test_train_generator_batch_shape(self):
        from utils.augmentation import get_train_generator
        import tensorflow as tf
        X = np.random.rand(100, 48, 48, 1).astype(np.float32)
        y = tf.keras.utils.to_categorical(np.random.randint(0, 7, 100), 7)
        gen = get_train_generator(X, y, batch_size=16)
        batch_X, batch_y = next(gen)
        assert batch_X.shape == (16, 48, 48, 1)
        assert batch_y.shape == (16, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
