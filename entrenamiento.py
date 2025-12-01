# ==============================================================
# Entrenamiento de modelo U-Net para segmentación de cabeza fetal
# Dataset: HC18 (Head Circumference Challenge)
# Autor: Fernando Godoy
# ==============================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Eliminado uso de scikit-learn para evitar sobrecarga en import
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_PATH = r'C:\Users\ferna\OneDrive\Escritorio\PROYECTO OBSTETRICIA\DATASET HC18'
MODEL_PATH = r'C:\Users\ferna\OneDrive\Escritorio\PROYECTO OBSTETRICIA\unet_hc18_best.h5'
METADATA_CSV = r'C:\Users\ferna\OneDrive\Escritorio\PROYECTO OBSTETRICIA\training_set_pixel_size_and_HC.csv'


# Configuración para optimizar rendimiento
print("Configurando TensorFlow para optimizar rendimiento...")
print(f"TensorFlow version: {tf.__version__}")

# Configurar para usar todos los núcleos de CPU disponibles
tf.config.threading.set_inter_op_parallelism_threads(0)  # Usar todos los núcleos
tf.config.threading.set_intra_op_parallelism_threads(0)  # Usar todos los núcleos

# Verificar dispositivos disponibles
print("Dispositivos disponibles:")
for device in tf.config.list_physical_devices():
    print(f"- {device}")

# Configurar para mejor rendimiento en CPU
tf.config.optimizer.set_jit(True)  # Habilitar XLA JIT compilation

# ==============================================================
# CONFIGURACIÓN INICIAL
# ==============================================================

# Ruta base donde tienes el dataset localmente (unificada)
TRAIN_PATH = os.path.join(BASE_PATH, "training_set")

# Parámetros
IMG_SIZE = 256
SEED = 42

# ==============================================================
# CARGA Y PREPROCESAMIENTO DE DATOS
# ==============================================================

def load_data(images_path):
    """
    Carga imágenes y máscaras del dataset HC18.
    Asume que las máscaras terminan con '_Annotation.png'.
    """
    images = sorted([f for f in os.listdir(images_path) if f.endswith(".png") and "Annotation" not in f])
    masks = sorted([f for f in os.listdir(images_path) if f.endswith("_Annotation.png")])

    X, Y = [], []
    for img_name, mask_name in zip(images, masks):
        img = cv2.imread(os.path.join(images_path, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(images_path, mask_name), cv2.IMREAD_GRAYSCALE)

        # Redimensionar
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Normalizar
        img = img / 255.0
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        Y.append(mask)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = np.array(Y)

    print(f"Imagenes cargadas: {len(X)}")
    return X, Y


# Cargar el dataset
X, Y = load_data(TRAIN_PATH)

# División entrenamiento/validación sin scikit-learn
def split_train_val(X, Y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X) * test_size)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]

X_train, X_val, Y_train, Y_val = split_train_val(X, Y, test_size=0.2, seed=SEED)
print(f"Conjunto entrenamiento: {X_train.shape}, validacion: {X_val.shape}")

# ==============================================================
# MODELO U-NET LIGERA
# ==============================================================

def unet_light(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)

    # ---- Encoder ----
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    # ---- Bottleneck ----
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c4)

    # ---- Decoder ----
    u5 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# Crear y compilar modelo
model = unet_light()
model.compile(optimizer=optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# ==============================================================
# ENTRENAMIENTO
# ==============================================================

checkpoint = ModelCheckpoint("unet_hc18_best.h5", monitor="val_loss",
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Configuración optimizada para CPU
batch_size = 16  # Aumentar batch size para mejor rendimiento
print(f"Usando batch size optimizado: {batch_size}")

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=batch_size,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ==============================================================
# VISUALIZACIÓN DE RESULTADOS
# ==============================================================

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución de la pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Evolución de la precisión')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ==============================================================
# PREDICCIÓN DE EJEMPLOS
# ==============================================================

idx = np.random.randint(0, len(X_val))
sample_img = X_val[idx]
sample_mask = Y_val[idx]
pred_mask = model.predict(np.expand_dims(sample_img, axis=0))[0]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title('Ecografía')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sample_mask.squeeze(), cmap='gray')
plt.title('Máscara Real')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(pred_mask.squeeze(), cmap='gray')
plt.title('Predicción IA')
plt.axis('off')
plt.show()
