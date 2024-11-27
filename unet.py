import os
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, UnidentifiedImageError
import numpy as np
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ora puoi usare le variabili d'ambiente nel tuo codice
PERCORSO_TRAINING_IMMAGINI = os.getenv('PERCORSO_TRAINING_IMMAGINI', '/Users/utente/Downloads/Training-labeled/images')
PERCORSO_TRAINING_LABELS = os.getenv('PERCORSO_TRAINING_LABELS', '/Users/utente/Downloads/Training-labeled/labels')
PERCORSO_TUNING_IMMAGINI = os.getenv('PERCORSO_TUNING_IMMAGINI', '/Users/utente/Downloads/Tuning/images')
PERCORSO_TUNING_LABELS = os.getenv('PERCORSO_TUNING_LABELS', '/Users/utente/Downloads/Tuning/labels')
PERCORSO_OUTPUT = os.getenv('PERCORSO_OUTPUT', '/Users/utente/Desktop/output-principi')

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
def load_grayscale_images(directory):
    images = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith('.tiff'):
            try:
                img = Image.open(os.path.join(directory, filename)).convert('L')
                img = img.resize((256, 256))
                images.append(np.array(img))
                filenames.append(filename)
            except (IOError, UnidentifiedImageError) as e:
                print(f"Errore nel caricamento dell'immagine {filename}: {e}")
    return np.array(images), filenames

# Funzione per caricare le etichette in formato TIFF corrispondenti alle immagini trovate
def load_tiff_labels(directory, filenames):
    labels = []
    for filename in filenames:
        label_filename = filename.replace('.tiff', '_label.tiff')
        try:
            img = Image.open(os.path.join(directory, label_filename))
            img = img.resize((256, 256))
            labels.append(np.array(img))
        except (IOError, UnidentifiedImageError) as e:
            print(f"Errore nel caricamento dell'etichetta {label_filename}: {e}")
    return np.array(labels)

# Carica le immagini e le etichette
def load_dataset(image_dir, label_dir):
    images, filenames = load_grayscale_images(image_dir)
    labels = load_tiff_labels(label_dir, filenames)
    print(f"Numero di immagini caricate: {len(images)}")
    print(f"Numero di etichette caricate: {len(labels)}")
    assert len(images) == len(labels), "Il numero di immagini e etichette non corrisponde."
    return tf.data.Dataset.from_tensor_slices((images, labels))

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
            pred = predictions[j].squeeze()  # Rimuovi dimensioni inutili
            img = Image.fromarray((pred * 255).astype(np.uint8), mode='L')
            img.save(os.path.join(output_dir, f"prediction_{i}_{j}.jpeg"))

# Salva le predizioni del set di tuning
save_predictions(tuning_dataset.batch(32), model, PERCORSO_OUTPUT)

# Visualizzazione del modello
model.summary()