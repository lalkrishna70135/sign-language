# 🖐️ Sign Language Alphabet Classifier (ASL A–Z)

> Real-time ASL **alphabet** gesture recognition using a Convolutional Neural Network (CNN), OpenCV, and TensorFlow — with training and live demo notebooks.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)  

---

## ✨ Highlights
- Trained a **CNN** to classify **ASL alphabet gestures (A–Z)** with high accuracy.  
- Built a **real-time prediction pipeline** using **OpenCV + TensorFlow**.  
- Includes two notebooks:  
  - **`colab.ipynb`** → end-to-end training (dataset loading, preprocessing, training, evaluation).  
  - **`live.ipynb`** → live webcam demo for real-time recognition.  
- Exported trained model: **`1.h5`**.  

> **Note:** This project currently recognizes **single alphabet signs only** — it does *not* perform full ASL sentence translation.  

---

## 🗂️ Repository Structure
```
.
├── 1.h5              # Trained CNN model
├── colab.ipynb       # Training notebook (end-to-end)
├── live.ipynb        # Live webcam demo
├── requirements.txt  # Dependencies
└── README.md         # Project description
```

---

## 🚀 Quickstart

### 1) Clone the repo
```bash
git clone https://github.com/<your-username>/sign-language.git
cd sign-language
```

### 2) Set up environment
```bash
# (Optional but recommended) create a virtual environment
python -m venv .venv
# Windows: .venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3) Train the model (optional)
Open **`colab.ipynb`** in Jupyter/Colab:  
- Update dataset paths.  
- Run all cells to train the CNN model.  
- Save trained weights (`.h5`).  

### 4) Run live demo
Open **`live.ipynb`** in Jupyter/Colab:  
- Activates webcam.  
- Classifies ASL alphabet gestures.  
- Displays predicted letter + confidence on video feed.  

---

## 📦 Requirements
`requirements.txt`
```txt
tensorflow>=2.9
opencv-python
numpy
scikit-learn
matplotlib
```

---

## 🧠 Model & Training
- **Architecture:** Custom CNN (Conv + Pooling layers → Dense layers).  
- **Loss/Optimizer:** Categorical Cross-Entropy + Adam.  
- **Augmentations:** Basic preprocessing applied in training notebook.  
- **Labels:** 26 output classes (A–Z).  

---

## 📊 Results
- Achieved high accuracy (~90–95%) on held-out test set (ASL alphabet dataset).  
- Confusion matrix and per-class metrics can be generated in `colab.ipynb`.  

---

## 🗺️ Roadmap
- [ ] Improve generalization with more augmentation.  
- [ ] Expand to words/sequences (RNN/Transformer for temporal modeling).  
- [ ] Convert model to **TF Lite** for mobile/edge deployment.  
- [ ] Add GUI (e.g., Streamlit/Tkinter).  

---

## 🤝 Contributing
Contributions are welcome!  
- Fork the repo and make your changes.  
- Open a pull request with a clear description of improvements.  

---

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.  

---

## 🙏 Acknowledgments
- TensorFlow & OpenCV communities.  
- Public ASL datasets used for training.  

---

## 🔖 Citation
If you use this repo in your research or project:  
```bibtex
@software{asl_alphabet_classifier_2023,
  title        = {Sign Language Alphabet Classifier (ASL A–Z)},
  author       = {lalkrishna},
  year         = {2023},
  url          = {https://github.com/lalkrishna70135/sign-language}
}
```
