import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from functools import partial  # Importa partial
import os
from dotenv import load_dotenv
import numpy as np

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ora puoi usare le variabili d'ambiente nel tuo codice
TRAINING = os.getenv('PERCORSO_TRAINING_LABELED')
TUNING = os.getenv('PERCORSO_TUNING')
TESTING = os.getenv('PERCORSO_TESTING')
OUTPUT = os.getenv('PERCORSO_OUTPUT')

# 1. Definire la CNN
def crea_modelo():
    modello = models.Sequential()

    # Primo layer convoluzionale
    modello.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    modello.add(layers.MaxPooling2D((2, 2)))

    # Secondo layer convoluzionale
    modello.add(layers.Conv2D(64, (3, 3), activation='relu'))
    modello.add(layers.MaxPooling2D((2, 2)))

    # Terzo layer convoluzionale
    modello.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten e fully connected layer
    modello.add(layers.Flatten())
    modello.add(layers.Dense(64, activation='relu'))

    # Output layer (numero di classi nel dataset)
    modello.add(layers.Dense(10, activation='softmax'))  # Cambia 10 se hai un numero diverso di classi

    # Compilazione del modello
    modello.compile(optimizer='adam',
                    loss='categorical_crossentropy',  # Usato per classificazione multi-classe
                    metrics=['accuracy'])

    return modello

# Funzione per caricare e gestire le immagini .tif
def load_image_tiff(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # Converte in RGB se non lo è
        return img
    except Exception as e:
        print(f"Errore nel caricare l'immagine {image_path}: {e}")
        return None

# Personalizzazione di ImageDataGenerator per caricare e gestire le immagini .tif
class TiffImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        # Usa il metodo originale per ottenere il flusso di file
        flow = super().flow_from_directory(directory, *args, **kwargs)

        # Modifica il flusso di file per usare il caricamento personalizzato
        flow.filenames = [f for f in flow.filenames if f.endswith('.tif') or f.endswith('.tiff')]

        # Crea una versione personalizzata di load_img
        custom_load_img = partial(self.custom_load_img, flow=flow)
        
        # Modifica il metodo di caricamento delle immagini
        load_img = custom_load_img  # Usa la funzione modificata
        return flow

    def custom_load_img(self, path, flow, *args, **kwargs):
        # Se l'immagine è TIFF, usa la funzione personalizzata
        if path.endswith('.tif') or path.endswith('.tiff'):
            return load_image_tiff(path)
        else:
            # Usa la funzione di caricamento immagine standard
            return load_img(path, *args, **kwargs)

# 2. Preparazione dei dati (Immagini e etichette)
train_dir = TRAINING  # Sostituisci con il percorso della tua cartella di training
test_dir = TESTING    # Sostituisci con il percorso della tua cartella di test

# Crea un TiffImageDataGenerator per la normalizzazione delle immagini
train_datagen = TiffImageDataGenerator(rescale=1./255)
test_datagen = TiffImageDataGenerator(rescale=1./255)

# Creazione dei generatori di dati per il training e il test
train_generator = train_datagen.flow_from_directory(
    train_dir,  # La cartella con le immagini di training
    target_size=(32, 32),  # Ridimensiona le immagini (32x32 per CIFAR-10 o adattato per il tuo dataset)
    batch_size=64,
    class_mode='categorical'  # Usa 'categorical' per classificazione multi-classe
)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # La cartella con le immagini di test
    target_size=(32, 32),  # Ridimensiona le immagini (32x32 per CIFAR-10 o adattato per il tuo dataset)
    batch_size=64,
    class_mode='categorical'  # Usa 'categorical' per classificazione multi-classe
)

# 3. Creazione del modello
modello = crea_modelo()

# 4. Allenamento del modello
modello.fit(
    train_generator,  # Generatore di dati per il training
    epochs=10,        # Numero di epoche (modifica secondo necessità)
    validation_data=test_generator  # Generatore di dati per la validazione
)

# 5. Valutazione del modello
test_loss, test_acc = modello.evaluate(test_generator)
print(f"Loss: {test_loss}, Accuracy: {test_acc}")

# 6. Salvataggio del modello
modello.save(OUTPUT + '/modello.h5')  # Salva il modello allenato

# 7. Caricamento del modello (opzionale, se necessario)
modello_caricato = tf.keras.models.load_model(OUTPUT + '/modello.h5')