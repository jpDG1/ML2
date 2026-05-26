import matplotlib
matplotlib.use('Agg')  # Zapis do pliku zamiast okna GUI

import os
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                     BatchNormalization, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR     = "data"
IMG_SIZE     = (48, 48)
BATCH_SIZE   = 32
EPOCHS       = 30
RANDOM_STATE = 42
NUM_CLASSES  = 7

# Data Pipeline
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=15,
    width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=True, seed=RANDOM_STATE
)
test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False
)

print(f"Klasy: {train_generator.class_indices}")
print(f"Liczba próbek treningowych: {train_generator.samples}")
print(f"Liczba próbek testowych:    {test_generator.samples}")

# Model bazowy
model_base = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2), Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(), Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
], name="model_bazowy")

model_base.summary()
model_base.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_base = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath='best_model_base.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n=== Trenowanie modelu bazowego ===\n")
history_base = model_base.fit(
    train_generator, epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks_base, verbose=1
)

# Wykresy modelu bazowego
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history_base.history['accuracy'],     label='Train')
ax1.plot(history_base.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy — model bazowy')
ax1.set_xlabel('Epoka'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history_base.history['loss'],     label='Train')
ax2.plot(history_base.history['val_loss'], label='Val')
ax2.set_title('Loss — model bazowy')
ax2.set_xlabel('Epoka'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle("Krzywe uczenia — model bazowy")
plt.tight_layout()
plt.savefig('cnn_krzywe_bazowy.png', dpi=100, bbox_inches='tight')
plt.close()
print("Zapisano: cnn_krzywe_bazowy.png")

base_loss, base_acc = model_base.evaluate(test_generator, verbose=1)
print(f"\nModel bazowy — Test Accuracy: {base_acc:.4f}  Loss: {base_loss:.4f}")

# Keras Tuner — Hyperband
print("\n=== Tuning hiperparametrów — Keras Tuner (Hyperband) ===\n")

def build_model(hp):
    filters_1   = hp.Choice('filters_blok1',  values=[32, 64])
    filters_2   = hp.Choice('filters_blok2',  values=[64, 128])
    drop_conv   = hp.Float ('dropout_conv',   min_value=0.1, max_value=0.4, step=0.1)
    dense_units = hp.Choice('dense_units',    values=[128, 256, 512])
    drop_dense  = hp.Float ('dropout_dense',  min_value=0.3, max_value=0.6, step=0.1)
    lr          = hp.Choice('learning_rate',  values=[1e-2, 1e-3, 1e-4])

    m = Sequential([
        Conv2D(filters_1, (3,3), padding='same', activation='relu', input_shape=(48,48,1)),
        BatchNormalization(),
        Conv2D(filters_1, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2), Dropout(drop_conv),

        Conv2D(filters_2, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(filters_2, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2), Dropout(drop_conv),

        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2), Dropout(min(drop_conv + 0.1, 0.5)),

        Flatten(),
        Dense(dense_units, activation='relu'),
        BatchNormalization(), Dropout(drop_dense),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m

tuner = kt.Hyperband(
    build_model, objective='val_accuracy',
    max_epochs=15, factor=3,
    directory='keras_tuner_dir',
    project_name='fer2013_cnn_tuning',
    overwrite=True
)
tuner.search_space_summary()
stop_early = EarlyStopping(monitor='val_loss', patience=5)
print("\n--- Przeszukiwanie przestrzeni hiperparametrów ---\n")
tuner.search(train_generator, validation_data=test_generator,
             callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nNajlepsza konfiguracja:")
print(f"  Filtry blok 1:      {best_hps.get('filters_blok1')}")
print(f"  Filtry blok 2:      {best_hps.get('filters_blok2')}")
print(f"  Dropout conv:       {best_hps.get('dropout_conv'):.1f}")
print(f"  Neurony Dense:      {best_hps.get('dense_units')}")
print(f"  Dropout Dense:      {best_hps.get('dropout_dense'):.1f}")
print(f"  Learning rate:      {best_hps.get('learning_rate')}")

print("\n=== Trenowanie najlepszego modelu ===\n")
model_tuned = tuner.hypermodel.build(best_hps)
callbacks_tuned = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath='best_model_tuned.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]
history_tuned = model_tuned.fit(
    train_generator, epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks_tuned, verbose=1
)

# Wykresy modelu po tuningu
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history_tuned.history['accuracy'],     label='Train')
ax1.plot(history_tuned.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy — model po tuningu')
ax1.set_xlabel('Epoka'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history_tuned.history['loss'],     label='Train')
ax2.plot(history_tuned.history['val_loss'], label='Val')
ax2.set_title('Loss — model po tuningu')
ax2.set_xlabel('Epoka'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle("Krzywe uczenia — model po tuningu Hyperband")
plt.tight_layout()
plt.savefig('cnn_krzywe_tuned.png', dpi=100, bbox_inches='tight')
plt.close()
print("Zapisano: cnn_krzywe_tuned.png")

tuned_loss, tuned_acc = model_tuned.evaluate(test_generator, verbose=1)

print("\n" + "="*50)
print("  PODSUMOWANIE PORÓWNAWCZE")
print("="*50)
print(f"  Model bazowy     | Accuracy: {base_acc:.4f} | Loss: {base_loss:.4f}")
print(f"  Model po tuningu | Accuracy: {tuned_acc:.4f} | Loss: {tuned_loss:.4f}")
poprawa = tuned_acc - base_acc
print(f"  Zmiana:          | {'+' if poprawa>=0 else ''}{poprawa:.4f}")
print("="*50)
print("\n=== Zajęcia 2 - Deep Learning zakończone! ===")