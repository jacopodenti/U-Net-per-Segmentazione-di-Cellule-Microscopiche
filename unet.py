import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, Dropout
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ora puoi usare le variabili d'ambiente nel tuo codice
TRAINING = os.getenv('PERCORSO_TRAINING_LABELED')
TUNING = os.getenv('PERCORSO_TUNING')
TESTING = os.getenv('PERCORSO_TESTING')
OUTPUT = os.getenv('PERCORSO_OUTPUT')

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck
    c5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(c9)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

train_dataset = image_dataset_from_directory(
    TRAINING,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    TUNING,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)
def preprocess_labels(dataset):
    def process(image, label):
        # Espandi le dimensioni di 'label' per includere il canale
        label = tf.expand_dims(label, axis=-1)  # (batch_size, height, width) -> (batch_size, height, width, 1)
        label = tf.image.resize(label, (256, 256))  # Ridimensiona l'etichetta a 256x256
        
        # Ridimensiona anche l'immagine
        image = tf.image.resize(image, (256, 256))
        
        # Stampa per il debug
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        
        return image, label
    return dataset.map(process)

train_dataset = preprocess_labels(train_dataset)
val_dataset = preprocess_labels(val_dataset)

# Compilazione del modello
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Funzione per salvare le predizioni come immagini
def save_predictions(dataset, model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for images, _ in dataset:
        predictions = model.predict(images)
        for i in range(len(images)):
            img = array_to_img(predictions[i])
            img.save(f"{output_dir}/prediction_{i}.png")

# Salva le predizioni del set di validazione
save_predictions(val_dataset, model, OUTPUT)

# Visualizzazione del modello
model.summary()
