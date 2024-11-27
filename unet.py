import os
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

# Funzione per costruire il modello UNet
def unet_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Funzione per caricare le immagini in scala di grigi
def load_grayscale_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    filenames = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.tiff', '.tif')):
            label_filename = filename.replace('.tiff', '_label.tiff').replace('.tif', '_label.tiff')
            label_path = os.path.join(label_dir, label_filename)
            if os.path.exists(label_path):
                try:
                    img = Image.open(os.path.join(image_dir, filename)).convert('L')
                    img = img.resize((256, 256))
                    img = np.array(img) / 255.0  # Normalizza i valori dei pixel tra 0 e 1
                    label = Image.open(label_path).convert('L')
                    label = label.resize((256, 256))
                    label = np.array(label) / 255.0  # Normalizza i valori dei pixel tra 0 e 1
                    images.append(img)
                    labels.append(label)
                    filenames.append(filename)
                except (IOError, UnidentifiedImageError) as e:
                    print(f"Errore nel caricamento dell'immagine o dell'etichetta {filename}: {e}")
    return np.array(images), np.array(labels), filenames

# Carica le immagini e le etichette
def load_dataset(image_dir, label_dir):
    images, labels, filenames = load_grayscale_images_and_labels(image_dir, label_dir)
    print(f"Numero di immagini caricate: {len(images)}")
    print(f"Numero di etichette caricate: {len(labels)}")
    assert len(images) == len(labels), "Il numero di immagini e etichette non corrisponde."
    return tf.data.Dataset.from_tensor_slices((images, labels))

# Imposta i percorsi direttamente nel codice
PERCORSO_TRAINING_IMMAGINI = '/Users/utente/Downloads/Training-labeled/images'
PERCORSO_TRAINING_LABELS = '/Users/utente/Downloads/Training-labeled/labels'
PERCORSO_TUNING_IMMAGINI = '/Users/utente/Downloads/Tuning/images'
PERCORSO_TUNING_LABELS = '/Users/utente/Downloads/Tuning/labels'
PERCORSO_OUTPUT = '/Users/utente/Desktop/output-principi'

# Carica i dataset di addestramento e tuning
train_dataset = load_dataset(PERCORSO_TRAINING_IMMAGINI, PERCORSO_TRAINING_LABELS)
tuning_dataset = load_dataset(PERCORSO_TUNING_IMMAGINI, PERCORSO_TUNING_LABELS)

# Costruisci e compila il modello
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestra il modello
model.fit(train_dataset.batch(32), validation_data=tuning_dataset.batch(32), epochs=10)

# Funzione per salvare le predizioni come immagini JPEG
def save_predictions(dataset, model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (images, _) in enumerate(dataset):
        predictions = model.predict(images)
        for j in range(len(images)):
            pred = (predictions[j] > 0.5).astype(np.uint8)  # Binarizza le predizioni
            img = Image.fromarray(pred.squeeze() * 255, mode='L')
            img.save(os.path.join(output_dir, f"prediction_{i}_{j}.jpeg"))

# Salva le predizioni del set di tuning
save_predictions(tuning_dataset.batch(32), model, PERCORSO_OUTPUT)

# Visualizzazione del modello
model.summary()

# Visualizza alcune predizioni per verifica
def visualize_predictions(dataset, model):
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(min(5, len(images))):
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(images[i].squeeze(), cmap='gray')  # Rimuovi .numpy() qui
            plt.subplot(1, 3, 2)
            plt.title("Predicted Mask")
            plt.imshow((predictions[i] > 0.5).astype(np.uint8).squeeze(), cmap='gray')  # Binarizza le predizioni
            plt.subplot(1, 3, 3)
            plt.title("Ground Truth")
            plt.imshow(labels[i].squeeze(), cmap='gray')  # Rimuovi .numpy() qui
            plt.show()

# Visualizza alcune predizioni del set di tuning
visualize_predictions(tuning_dataset.batch(32), model)