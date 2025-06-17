# pip install tensorflow
# pip install opencv-python
# if you also need additional OpenCV functionalities (like video processing), install: opencv-python-headless
# pip install opencv-python-headless
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, fbeta_score
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

DATASET_PATH = "fer2013.csv"  # Ensure the file is in the working directory
df = pd.read_csv(DATASET_PATH)

# Extract data
X, y = [], []
IMG_SIZE = 48  # FER-2013 images are 48x48 pixels
num_classes = 7  # FER-2013 has 7 emotion labels

for index, row in df.iterrows():
    img = np.array(row['pixels'].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
    X.append(img)
    y.append(row['emotion'])

# Convert to numpy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
y = to_categorical(y, num_classes=num_classes)  # One-hot encode labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6, batch_size=32)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Save model
model.save("fer2013_emotion_model.h5")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# SHAP Visualizations
import shap

def shap_visualizations(model, X_sample, class_names=None):
    """
    Compute and visualize SHAP explanations for a CNN model on image data.
    
    Parameters:
    - model: trained Keras model
    - X_sample: small sample of input images (N, 48, 48, 1)
    - class_names: list of class labels (optional)
    """
    # Use a small background sample to minimize warnings
    background = X_sample[:5]
    
    # Try DeepExplainer first
    try:
        explainer = shap.DeepExplainer(model, background)
    except Exception as e:
        print(f"[SHAP] DeepExplainer failed: {e}")
        print("[SHAP] Falling back to GradientExplainer...")
        explainer = shap.GradientExplainer(model, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Visualize per-class image importance maps
    print("[SHAP] Displaying image_plot (per class)")
    shap.image_plot(shap_values, X_sample)

# ----------------- SHAP Visualisation -----------------
X_sample = X_test[:3]
# Optional: Class labels if you have them
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
shap_visualizations(model, X_sample, class_names)

# Hyperparameter tuning
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
    for _ in range(hp['num_layers']):
        model.add(layers.Conv2D(hp['neurons'], (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        if hp['dropout_rate'] > 0:
            model.add(layers.Dropout(hp['dropout_rate']))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp['dense_neurons'], activation='relu'))
    if hp['dropout_rate'] > 0:
        model.add(layers.Dropout(hp['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp['learning_rate']) if hp['optimizer'] == 'adam' else tf.keras.optimizers.SGD(learning_rate=hp['learning_rate'])
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Random Search + Grid Search
def hyperparameter_search(X_train, y_train, X_val, y_val, num_random=5):
    # Random search space
    search_space = {
        'optimizer': ['adam', 'sgd'],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [32, 64],
        'num_layers': [1, 2],
        'neurons': [32, 64],
        'dense_neurons': [64],
        'dropout_rate': [0.3, 0.5]
    }
    
    # Random sampling
    random_samples = [
        {key: random.choice(values) for key, values in search_space.items()}
        for _ in range(num_random)
    ]
    
    best_accuracy = 0
    best_hp = None
    best_model = None
    
    for idx, hp in enumerate(random_samples):
        print(f"Training random search model {idx+1}/{num_random} with params: {hp}")
        model = build_model(hp)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=hp['batch_size'], verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_hp = hp
            best_model = model
    
    print(f"Best Random Search Params: {best_hp}")

    # Fine Grid Search around the best random sample
    grid_space = {
        'learning_rate': [best_hp['learning_rate'] * 0.5, best_hp['learning_rate'], best_hp['learning_rate'] * 1.5],
        'dropout_rate': [max(0, best_hp['dropout_rate'] - 0.1), best_hp['dropout_rate'], min(0.8, best_hp['dropout_rate'] + 0.1)]
    }
    
    fine_grid = list(ParameterGrid(grid_space))
    
    for idx, grid_params in enumerate(fine_grid):
        print(f"Training grid search model {idx+1}/{len(fine_grid)} with fine-tuned params: {grid_params}")
        hp_updated = best_hp.copy()
        hp_updated.update(grid_params)
        model = build_model(hp_updated)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=hp_updated['batch_size'], verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_hp = hp_updated
            best_model = model
    
    print(f"Best Fine-Tuned Params: {best_hp}")
    return best_model, best_hp

# Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    f2 = fbeta_score(y_true, y_pred_classes, beta=2, average='weighted')
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, F2-score: {f2:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # ROC Curve
    y_test_bin = label_binarize(y_true, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend()
    plt.show()

# ------------------ RUN ------------------
# Run hyperparameter search
best_model, best_hyperparams = hyperparameter_search(X_train, y_train, X_test, y_test)

# Evaluate final best model
evaluate_model(best_model, X_test, y_test)

# Save final model
best_model.save("best_fer2013_emotion_model.h5")