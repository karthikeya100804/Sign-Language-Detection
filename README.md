
# 🤟 Real-Time American Sign Language (ASL) Recognition

This project aims to recognize static American Sign Language (ASL) gestures in real-time using computer vision and machine learning. It includes both a training pipeline using keypoint extraction and a real-time detection system using a webcam.

## 📌 Project Highlights

-  Supports 29 ASL gestures (A-Z, `space`, `del`, `nothing`)
-  Real-time gesture recognition using webcam
-  Trained on a keypoint-based dataset extracted via MediaPipe
-  ML Models used: Neural Network, Random Forest, Gradient Boosting
-  Achieved ~96%+ accuracy on test set
-  Optimized for real-time performance and mobile deployment

---

## 📁 Project Structure

```
.
├── asl_realTime.py              # Real-time detection using webcam
├── ASL.ipynb                    # Jupyter Notebook for model training
├── asl_model.keras              # Trained Keras model for inference
├── asl_keypoints_dataset_1000.csv  # Dataset with extracted hand keypoints
└── README.md                    # You're reading this!
```

---

## 🧠 Model Training Pipeline (ASL.ipynb)

1. **Dataset:** ASL Alphabet Dataset from Kaggle (~87K images)
2. **Preprocessing:**
   - 1,000 images per class sampled
   - MediaPipe used to extract 21 hand landmarks → 63 keypoints
3. **Models:**
   - 🧠 Feedforward Neural Network (3 layers)
   - 🌲 Random Forest (100 trees)
   - 🔁 Gradient Boosting (100 estimators)
4. **Evaluation:**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix & performance graphs
5. **Result Summary:**

- Neural Network: 96.44% accuracy
- Random Forest: 96.84% accuracy
- Gradient Boosting: 94.76% accuracy

---

## 🎥 Real-Time Detection (asl_realTime.py)

### 🔧 How it Works:
- Captures live webcam frames using OpenCV
- Preprocesses frame to 128x128
- Uses trained model to classify hand gesture
- Displays label and confidence on-screen

### ▶️ Run it:

```bash
pip install opencv-python tensorflow numpy
python asl_realTime.py
```

### 🔄 Controls:
- Press `q` to quit

---

## 📦 Dependencies

- `tensorflow`
- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `plotly`

---

## 📂 Dataset Info

- **Name:** [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Classes:** 29 (A-Z, del, space, nothing)
- **Size:** ~87,000 images (200x200 px RGB)

---

## 📜 License

This project is licensed under the [GPL-2.0 License](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).

---


