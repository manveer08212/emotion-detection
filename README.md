# 🎭 Emotion Detection using Deep Learning

Real-time facial emotion detection system built with Python, TensorFlow, and OpenCV. CNN trained on FER-2013 dataset classifies **7 emotions** from live camera feeds.

> **Project Duration:** Jan 2024 – Jun 2024  
> **Accuracy Improvement:** +15% via data preprocessing & CNN fine-tuning

---

## 📁 Project Structure

```
emotion-detection/
├── src/
│   ├── train.py              # CNN training pipeline
│   ├── predict.py            # Real-time webcam inference
│   └── evaluate.py           # Model evaluation & metrics
├── utils/
│   ├── preprocessing.py      # FER-2013 data preprocessing
│   ├── augmentation.py       # Data augmentation utilities
│   └── visualization.py      # Plotting & confusion matrix
├── models/
│   └── emotion_cnn.py        # CNN architecture definition
├── notebooks/
│   └── exploration.ipynb     # EDA & training experiments
├── data/
│   ├── raw/                  # Place fer2013.csv here
│   └── processed/            # Auto-generated processed arrays
├── tests/
│   └── test_model.py         # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

Custom CNN trained on FER-2013:

| Layer | Details |
|-------|---------|
| Input | 48×48 grayscale |
| Conv Block 1 | 2× Conv2D(32) + BN + MaxPool + Dropout(0.25) |
| Conv Block 2 | 2× Conv2D(64) + BN + MaxPool + Dropout(0.25) |
| Conv Block 3 | 2× Conv2D(128) + BN + MaxPool + Dropout(0.25) |
| Dense | 256 units + Dropout(0.5) |
| Output | 7 units, Softmax |

**Emotions:** Angry 😠 | Disgust 🤢 | Fear 😨 | Happy 😊 | Sad 😢 | Surprise 😲 | Neutral 😐

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
```

### 2. Get Dataset

Download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle and place `fer2013.csv` in `data/raw/`.

### 3. Preprocess Data

```bash
python utils/preprocessing.py
```

### 4. Train Model

```bash
python src/train.py --epochs 50 --batch-size 64
```

### 5. Real-Time Detection

```bash
python src/predict.py
```
Press `q` to quit the webcam feed.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~68% |
| Baseline Accuracy | ~53% |
| Improvement | **+15%** |
| FPS (CPU) | ~20 |
| FPS (GPU) | ~60 |

---

## ⚙️ Configuration

Edit `src/train.py` top-level constants or pass CLI args:

| Arg | Default | Description |
|-----|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--model-path` | `models/best_model.keras` | Save path |

---

## 🛠️ Tech Stack

- **Python** 3.9+
- **TensorFlow / Keras** 2.13+
- **OpenCV** 4.8+
- **NumPy**, **Pandas**, **Matplotlib**, **scikit-learn**

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
