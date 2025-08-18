#  Sign Language Translator (ASL → English)

> Real‑time ASL gesture recognition using a Convolutional Neural Network (CNN), OpenCV, and TensorFlow — packaged with a simple interface for live predictions.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

---

## ✨ Highlights
- Trained a **CNN** to translate **ASL hand gestures** to English with **~95% accuracy** (evaluated on held‑out test set).
- Built a **real‑time prediction pipeline** with **OpenCV + TensorFlow**, including preprocessing (ROI extraction, normalization) and post‑processing (prediction smoothing).
- Delivered a **live webcam interface** for users to see predictions instantly, with optional bounding box/heatmap overlays.
- Managed the **full lifecycle**: dataset collection/curation, augmentation, training, evaluation, and deployment.
- Modular code: switch models (baseline CNN → MobileNet/ResNet) or datasets with minimal changes.

> **Project timeline:** Jul 2023 – Dec 2023

---

## 🗂️ Repository Structure (suggested)
```
.
├── app/
│   ├── live.py               # Live webcam prediction (OpenCV + TensorFlow)
│   ├── infer_image.py        # Single image inference
│   └── utils.py              # Common preprocessing/inference utilities
├── notebooks/
│   ├── colab.ipynb           # End‑to‑end training on Colab
│   └── live.ipynb            # Quick demo notebook for live predictions
├── models/
│   └── model.h5              # Trained TensorFlow/Keras model (placeholder)
├── data/
│   ├── raw/                  # Raw dataset (not tracked)
│   └── processed/            # Preprocessed/augmented data (not tracked)
├── requirements.txt
├── README.md
└── LICENSE
```
> Note: Large artifacts and raw data should be ignored via `.gitignore`. Consider using Git LFS or a hosted release for the trained model.

---

## 🚀 Quickstart

### 1) Set up environment
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Recommended) Create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Add a trained model
- Place your trained Keras/TensorFlow model at `models/model.h5`
- Or update the `MODEL_PATH` inside scripts to point to your file.

### 3) Run live webcam demo
```bash
python app/live.py
```
Press `q` to quit. You should see predicted labels overlayed on the video stream.

---

## 📦 Requirements

`requirements.txt`
```txt
tensorflow>=2.9
opencv-python
numpy
pandas
scikit-learn
matplotlib
```
> If you face GPU/driver issues, install the appropriate TensorFlow build for your hardware (refer to TF docs).

---

## 🧠 Model & Training

- **Architecture:** Baseline **CNN** (Keras Sequential) with convolution + pooling blocks, followed by dense layers.
- **Loss/Optimizer:** Categorical Cross‑Entropy + Adam.
- **Augmentations:** Random flip/rotation/brightness (configurable).
- **Evaluation:** Accuracy, confusion matrix, per‑class metrics.

### Train on Colab / locally
Use the notebook:
- `notebooks/colab.ipynb` – end‑to‑end training
- Update dataset paths and hyperparameters as needed.

**Typical training loop (Keras):**
```python
model = tf.keras.Sequential([...])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[...])
model.save("models/model.h5")
```

---

## 🎯 Inference

### Live (webcam)
`app/live.py` (minimal example):
```python
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/model.h5"
LABELS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]  # adjust to your classes
IMG_SIZE = 64  # match training

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(frame):
    # Define ROI or use full frame; keep consistent with training!
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)  # (H, W, 1) if model is grayscale
    x = np.expand_dims(x, axis=0)   # (1, H, W, C)
    return x

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Cannot open webcam"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    x = preprocess(frame)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = LABELS[idx]
    conf = float(probs[idx])

    # Draw overlay
    cv2.putText(frame, f"{label} ({conf:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("ASL Translator - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Single Image
```bash
python app/infer_image.py --image path/to/image.jpg
```

---

## 📊 Results
- **Top‑1 Accuracy:** ~**95%** (test set; ASL gesture classification)
- Include your **confusion matrix**, **per‑class F1**, and sample predictions in this section.
- (Optional) Add a short discussion of failure cases and planned fixes.

> Tip: Export plots directly from your training notebook and embed them here (`/assets/plots/`).

---

## 🧱 Design Notes
- Consistent preprocessing (color → grayscale, normalization, target size) between training & inference.
- Lightweight CNN for fast inference on CPU; easy to swap a backbone (e.g., MobileNetV2) for improved accuracy.
- Simple label mapping allows quick extension to more classes/words.

---

## 🗺️ Roadmap
- [ ] Add temporal smoothing (sliding window majority vote).
- [ ] Expand vocabulary (beyond alphabet) with multi‑label sequences.
- [ ] Export to **TF Lite** for mobile/edge deployment.
- [ ] Optional GUI using **Tkinter**/**Streamlit**.
- [ ] Dockerfile and GitHub Actions CI.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue to discuss substantial changes. \
Follow conventional commits and include before/after metrics for model changes.

---

## 📜 License
This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgments
- OpenCV and TensorFlow communities.
- ASL learning resources and open datasets.
- Everyone who contributed feedback and testing.

---

## 🔖 Citation
If you use this repo in your research or project:
```bibtex
@software{asl_translator_2023,
  title        = {Sign Language Translator (ASL → English)},
  author       = {<Your Name>},
  year         = {2023},
  url          = {https://github.com/<your-username>/<your-repo>}
}
```
