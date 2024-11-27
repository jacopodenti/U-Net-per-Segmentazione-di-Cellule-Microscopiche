import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from dotenv import load_dotenv
# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Path ai dataset
training_images_path = os.getenv('TRAINING_IMAGES_PATH')
training_labels_path = os.getenv('TRAINING_LABELS_PATH')
tuning_images_path = os.getenv('TUNING_IMAGES_PATH')
tuning_labels_path = os.getenv('TUNING_LABELS_PATH')

# Funzione per raccogliere file con una specifica estensione
def collect_files(base_path, file_extension=".tif"):
    """
    Raccoglie tutti i file con una specifica estensione da una directory e sottodirectory.
    """
    files = []
    for root, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

# Funzione per caricare immagini TIFF a 64-bit con Pillow
def load_tiff_image(image_path):
    """
    Carica un'immagine TIFF utilizzando Pillow e converte a 8-bit.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("L")  # Converte in scala di grigi (8-bit)
        return np.array(img)  # Converte in array NumPy
    except Exception as e:
        raise ValueError(f"Errore nel caricamento dell'immagine TIFF: {image_path}. Dettagli: {e}")

# Funzione per applicare il filtro di thresholding
def apply_threshold(image, threshold=128):
    """
    Applica il filtro di thresholding a un'immagine.
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# Funzione per preparare il dataset
def prepare_dataset(image_paths, label_paths, threshold=128, image_size=(128, 128)):
    """
    Prepara il dataset trasformando immagini e label in array numpy.
    """
    images = []
    labels = []
    
    for img_path, lbl_path in zip(image_paths, label_paths):
        # Carica e processa l'immagine
        img = load_tiff_image(img_path)  # Usa Pillow per caricare l'immagine
        binary_image = apply_threshold(img, threshold)
        binary_image = cv2.resize(binary_image, image_size)  # Ridimensiona

        # Carica e processa il file di label
        label = load_tiff_image(lbl_path)  # Usa Pillow per le label
        label = cv2.resize(label, image_size)

        images.append(binary_image)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Funzione per creare una CNN
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

# Raccolta dei file
training_images = collect_files(training_images_path, file_extension=".tif")
training_labels = collect_files(training_labels_path, file_extension=".tiff")
tuning_images = collect_files(tuning_images_path, file_extension=".tif")
tuning_labels = collect_files(tuning_labels_path, file_extension=".tiff")

# Preparazione del dataset
X_train, y_train = prepare_dataset(training_images, training_labels)
X_tuning, y_tuning = prepare_dataset(tuning_images, tuning_labels)

# Normalizzazione e one-hot encoding
X_train = X_train[..., np.newaxis] / 255.0  # Aggiunge un asse per i canali
X_tuning = X_tuning[..., np.newaxis] / 255.0

# Conversione delle etichette a one-hot encoding
y_train = to_categorical(y_train)
y_tuning = to_categorical(y_tuning)

# Creazione del modello
input_shape = X_train.shape[1:]  # Esempio: (128, 128, 1)
num_classes = y_train.shape[-1]  # Numero di classi
model = create_cnn(input_shape, num_classes)

# Compilazione e addestramento del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_tuning, y_tuning))

# Valutazione del modello
loss, accuracy = model.evaluate(X_tuning, y_tuning)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Test del modello su una nuova immagine
def predict_image(model, image_path, threshold=128):
    """
    Predice la classe di una nuova immagine.
    """
    img = load_tiff_image(image_path)
    binary_image = apply_threshold(img, threshold)
    binary_image = cv2.resize(binary_image, input_shape[:2])
    binary_image = binary_image[np.newaxis, ..., np.newaxis] / 255.0
    
    prediction = model.predict(binary_image)
    return np.argmax(prediction)

# Test su un'immagine nuova
new_image_path = 'path/to/new_image.tif'
predicted_class = predict_image(model, new_image_path)
print(f"La classe prevista è: {predicted_class}")