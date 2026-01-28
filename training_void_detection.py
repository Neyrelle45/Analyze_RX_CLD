"""
Script d'entraînement pour la détection de voids et manques de soudure
À exécuter dans Google Colab
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from google.colab import drive
import json
from tqdm import tqdm

# Configuration des chemins
MYDRIVE_REAL = '/content/drive/MyDrive'
MYDRIVE_LINK = 'MyDrive'
ROOT_NAME = 'Analyze_RX'
SUBFOLDERS = ['rx_images', 'labels', 'masks', 'models', 'resultats']

# Monter Google Drive
drive.mount('/content/drive')

# Créer la structure de dossiers
root_path = os.path.join(MYDRIVE_REAL, ROOT_NAME)
for folder in SUBFOLDERS:
    os.makedirs(os.path.join(root_path, folder), exist_ok=True)

print(f"Structure créée dans: {root_path}")

# Chemins des dossiers
IMAGES_PATH = os.path.join(root_path, 'rx_images')
LABELS_PATH = os.path.join(root_path, 'labels')
MODELS_PATH = os.path.join(root_path, 'models')

class VoidDetectionDataGenerator(keras.utils.Sequence):
    """Générateur de données avec augmentation"""
    
    def __init__(self, image_paths, label_paths, batch_size=8, img_size=(512, 512), 
                 augment=True, shuffle=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[k] for k in indexes]
        batch_labels = [self.label_paths[k] for k in indexes]
        
        X, y = self._generate_data(batch_images, batch_labels)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _augment_image(self, image, mask):
        """Augmentation de données"""
        if not self.augment:
            return image, mask
        
        # Rotation aléatoire
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        
        # Flip horizontal
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Flip vertical
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Ajustement de luminosité et contraste
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contraste
            beta = np.random.uniform(-20, 20)     # Luminosité
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Ajout de bruit gaussien
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        return image, mask
    
    def _generate_data(self, batch_image_paths, batch_label_paths):
        X = np.empty((len(batch_image_paths), *self.img_size, 1), dtype=np.float32)
        y = np.empty((len(batch_image_paths), *self.img_size, 3), dtype=np.float32)
        
        for i, (img_path, label_path) in enumerate(zip(batch_image_paths, batch_label_paths)):
            # Charger l'image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            
            # Charger le label
            label = cv2.imread(label_path, cv2.IMREAD_COLOR)
            label = cv2.resize(label, self.img_size)
            
            # Augmentation
            img, label = self._augment_image(img, label)
            
            # Normalisation de l'image
            img = img.astype(np.float32) / 255.0
            
            # Créer les masques binaires:
            # Rouge (soudure): canal 0
            # Jaune (voids/manques): canal 1
            # Fond noir: canal 2
            mask_soudure = (label[:, :, 2] > 200) & (label[:, :, 1] < 100)  # Rouge
            mask_voids = (label[:, :, 2] > 200) & (label[:, :, 1] > 200)   # Jaune
            mask_fond = ~(mask_soudure | mask_voids)
            
            y_mask = np.stack([
                mask_soudure.astype(np.float32),
                mask_voids.astype(np.float32),
                mask_fond.astype(np.float32)
            ], axis=-1)
            
            X[i] = np.expand_dims(img, axis=-1)
            y[i] = y_mask
        
        return X, y


def build_unet_model(input_shape=(512, 512, 1), num_classes=3):
    """
    Construit un modèle U-Net optimisé pour la détection de voids
    Architecture légère mais efficace (<200MB)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Normalisation
    x = layers.BatchNormalization()(inputs)
    
    # Encoder
    # Block 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)
    
    # Block 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)
    
    # Block 3
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)
    
    # Block 4
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)
    
    # Bridge
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    
    # Decoder
    # Block 6
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.2)(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.2)(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Dropout(0.1)(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Dropout(0.1)(c9)
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice pour évaluer la segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Loss basée sur le coefficient de Dice"""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combinaison de categorical crossentropy et dice loss"""
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * cce + 0.5 * dice


def prepare_dataset():
    """Prépare le dataset d'entraînement"""
    image_files = sorted([f for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))])
    
    image_paths = []
    label_paths = []
    
    for img_file in image_files:
        img_path = os.path.join(IMAGES_PATH, img_file)
        
        # Chercher le fichier label correspondant
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '_label.png'
        label_path = os.path.join(LABELS_PATH, label_file)
        
        if os.path.exists(label_path):
            image_paths.append(img_path)
            label_paths.append(label_path)
    
    print(f"Nombre d'images trouvées: {len(image_paths)}")
    
    # Split train/validation
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_paths, label_paths, test_size=0.15, random_state=42
    )
    
    return train_imgs, val_imgs, train_labels, val_labels


def train_model(epochs=100, batch_size=4, img_size=(512, 512)):
    """Entraîne le modèle"""
    
    print("Préparation du dataset...")
    train_imgs, val_imgs, train_labels, val_labels = prepare_dataset()
    
    # Générateurs de données
    train_gen = VoidDetectionDataGenerator(
        train_imgs, train_labels, 
        batch_size=batch_size, 
        img_size=img_size, 
        augment=True
    )
    
    val_gen = VoidDetectionDataGenerator(
        val_imgs, val_labels, 
        batch_size=batch_size, 
        img_size=img_size, 
        augment=False
    )
    
    print(f"Images d'entraînement: {len(train_imgs)}")
    print(f"Images de validation: {len(val_imgs)}")
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_unet_model(input_shape=(*img_size, 1), num_classes=3)
    
    # Compiler
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    print(model.summary())
    
    # Callbacks
    model_path = os.path.join(MODELS_PATH, 'void_detection_best.h5')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(MODELS_PATH, 'void_detection_final.h5')
    model.save(final_model_path)
    
    print(f"\nModèle sauvegardé: {final_model_path}")
    
    # Sauvegarder l'historique
    history_path = os.path.join(MODELS_PATH, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    
    # Afficher les courbes d'apprentissage
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Affiche les courbes d'apprentissage"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Dice Coefficient
    axes[2].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[2].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_PATH, 'training_curves.png'), dpi=150)
    plt.show()


# Lancer l'entraînement
if __name__ == "__main__":
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE DE DÉTECTION DE VOIDS")
    print("="*60)
    
    model, history = train_model(
        epochs=100,
        batch_size=4,
        img_size=(512, 512)
    )
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ!")
    print("="*60)
