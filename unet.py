import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras import layers, models

# Carica le variabili d'ambiente dal file .env
load_dotenv()

TRAINING_IMAGES = os.getenv('PERCORSO_TRAINING_LABELED')
TRAINING_LABELS = os.getenv('PERCORSO_TRAINING_LABELED_LABELS')
TUNING_IMAGES = os.getenv('PERCORSO_TUNING')
TUNING_LABELS = os.getenv('PERCORSO_TUNING_LABELS')

def collect_files(base_path, file_extension=".tif"):
    """
    Raccolta di tutti i file con una specifica estensione da una directory e sottodirectory.
    """
    files = []
    for root, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

training_images = collect_files(TRAINING_IMAGES, file_extension=".tif")
training_labels = collect_files(TRAINING_LABELS, file_extension=".tiff")
tuning_images = collect_files(TUNING_IMAGES, file_extension=".tif")
tuning_labels = collect_files(TUNING_LABELS, file_extension=".tiff")

def apply_threshold(image_path, threshold=128):
    """
    Applica il filtro di thresholding a un'immagine.
    """
    # Carica l'immagine
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Applica il thresholding
    _, binary_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_image

test_image = training_images[0]  # Scegli un'immagine a caso
binary_img = apply_threshold(test_image)

# Visualizza l'immagine originale e quella binaria
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Originale")
plt.imshow(cv2.imread(test_image, cv2.IMREAD_GRAYSCALE), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Binary")
plt.imshow(binary_img, cmap="gray")
plt.show()


def prepare_dataset(image_paths, label_paths, threshold=128, image_size=(128, 128)):
    """
    Prepara il dataset trasformando immagini e label in array numpy.
    """
    images = []
    labels = []
    
    for img_path, lbl_path in zip(image_paths, label_paths):
        # Applica thresholding all'immagine
        binary_image = apply_threshold(img_path, threshold)
        binary_image = cv2.resize(binary_image, image_size)  # Ridimensiona
        
        # Carica il file di label e ridimensiona
        label_image = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        label_image = cv2.resize(label_image, image_size)
        
        images.append(binary_image)
        labels.append(label_image)
    
    return np.array(images), np.array(labels)

X_train, y_train = prepare_dataset(training_images, training_labels)
X_tuning, y_tuning = prepare_dataset(tuning_images, tuning_labels)

# Aggiungiamo un asse per i canali delle immagini (da 2D a 3D)
X_train = X_train[..., np.newaxis] / 255.0  # Normalizziamo
X_tuning = X_tuning[..., np.newaxis] / 255.0

def create_cnn(input_shape, num_classes):
    """
    Definisce e restituisce un modello CNN.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Definisci la forma degli input e il numero di classi
input_shape = X_train.shape[1:]  # Ad esempio (128, 128, 1)
num_classes = y_train.shape[-1]  # Numero di classi

model = create_cnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestramento
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_tuning, y_tuning))


loss, accuracy = model.evaluate(X_tuning, y_tuning)
print(f"Loss: {loss}, Accuracy: {accuracy}")

def predict_image(model, image_path, threshold=128):
    """
    Predice la classe di una nuova immagine.
    """
    binary_image = apply_threshold(image_path, threshold)
    binary_image = cv2.resize(binary_image, input_shape[:2])
    binary_image = binary_image[np.newaxis, ..., np.newaxis] / 255.0
    
    prediction = model.predict(binary_image)
    return np.argmax(prediction)

new_image = 'path/to/new_image.tif'
predicted_class = predict_image(model, new_image)
print(f"La classe prevista è: {predicted_class}")