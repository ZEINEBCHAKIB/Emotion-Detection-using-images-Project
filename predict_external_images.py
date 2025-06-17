import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("fer2013_emotion_model.h5")

# Emotion labels (consistent with FER-2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Image size expected by model
IMG_SIZE = 48

# --- List your image file names ---
external_image_paths = ["image1.jpeg", "image2.jpeg", "image3.jpeg", "image4.jpg"]

# --- Preprocess and load images ---
raw_images = []
preprocessed_images = []

for path in external_image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{path}' not found.")
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(IMG_SIZE, IMG_SIZE, 1)
    
    raw_images.append(img_resized)  # for display
    preprocessed_images.append(img_reshaped)

# Convert to numpy array
images_np = np.array(preprocessed_images)

# Predict emotions
predictions = model.predict(images_np)
predicted_classes = np.argmax(predictions, axis=1)

# Plot images with predicted labels
plt.figure(figsize=(15, 5))
for i in range(len(external_image_paths)):
    plt.subplot(1, len(external_image_paths), i+1)
    plt.imshow(raw_images[i], cmap='gray')
    plt.title(f"Predicted: {emotion_labels[predicted_classes[i]]}")
    plt.axis('off')

plt.suptitle("Emotion Predictions (External Images)", fontsize=16)
plt.show()