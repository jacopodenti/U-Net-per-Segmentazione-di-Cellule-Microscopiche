import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

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

# 2. Preparazione dei dati (Immagini e etichette)
train_dir = 'path_to_train_data'  # Sostituisci con il percorso della tua cartella di training
test_dir = 'path_to_test_data'    # Sostituisci con il percorso della tua cartella di test

# Crea un ImageDataGenerator per la normalizzazione delle immagini
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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
modello.save('modello.h5')  # Salva il modello allenato

# 7. Caricamento del modello (opzionale, se necessario)
modello_caricato = tf.keras.models.load_model('modello.h5')