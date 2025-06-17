#  Emotion Detection System using Deep Learning

---

# Overview
This project implements a deep learning-based emotion detection system using Convolutional Neural Networks (CNNs) applied to facial images. The system performs:
- Image preprocessing
- CNN model training
- Hyperparameter tuning (grid search, random search)
- Model evaluation using metrics like precision, recall, F1-score, ROC-AUC
- Model interpretability with SHAP visualizations

This work was developed as part of the course Introduction à l’Intelligence Artificielle (2024-2025), under Prof. Faouzi Marzouki, Ecole Supérieure de la Technologie, Tétouan.

---

## Objectives
- Learn to design and train CNN models for image classification.
- Apply computer vision techniques for emotion recognition.
- Analyze and interpret model performance.
- Connect AI systems with the physical environment through vision sensors.

---

## Files
- emotionDetection.py : Builds, trains, evaluates, and saves the CNN emotion detection model.
- predict_external_images.py : Tests the model on both FER-2013 dataset images and external images.
- FER-2013.csv : Dataset containing grayscale images of faces and their emotion labels.
- fer2013_emotion_model.h5 : Initial trained model before tuning.

---

# How to Run
## Train the model
**python emotionDetection.py (on bash)
This will:
- Load and preprocess data (normalization, one-hot encoding)
- Build a CNN model
- Train and save the model
- Perform hyperparameter tuning and save the best model
## Predict emotions on external images
** python predict_external_images.py (on bash)
You can modify this script to specify external image paths for prediction.
Update the file names in predict_external_images.py to match your images

---

# 📥 Dataset
The **FER-2013.csv** dataset is too large for this repository (287 MB).  
➡ **Download it manually from this link:**  
[Download FER-2013.csv](drive.google.com/file/d/10pbtcrBYyMrzuWGf5g_KAaWSYO60ED7v/view?usp=sharing)

 **Place `FER-2013.csv` in the same folder as your code files** (`emotionDetection.py`, etc).

---

# Installation
1️⃣ Clone this repository or download the code:
```bash
git clone https://github.com/ZEINEBCHAKIB/Emotion-detection-using-images.git
cd Emotion-detection-using-images 
*2️⃣ Install dependencies:
pip install tensorflow scikit-learn pandas numpy matplotlib opencv-python shap 
*3️⃣ Ensure you have the FER-2013.csv dataset in the working directory.

---

*# 👨‍💻 Contrubutors
*- Student: Zeineb Chakib
*- Professor: Faouzi Marzouki