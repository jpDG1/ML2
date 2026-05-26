import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PARAMETRY
# ============================================================
DATA_DIR = "data"
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
RANDOM_STATE = 42

# ============================================================
# 2. ANALIZA STRUKTURY DANYCH
# ============================================================
print("=== Analiza struktury danych FER-2013 ===\n")

# Zliczanie obrazów per klasa
splits = ["train", "test"]
class_counts = {}

for split in splits:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        print(f"Brak folderu: {split_path}")
        continue

    print(f"--- {split.upper()} ---")
    class_counts[split] = {}

    for emotion in sorted(os.listdir(split_path)):
        emotion_path = os.path.join(split_path, emotion)
        if os.path.isdir(emotion_path):
            count = len(os.listdir(emotion_path))
            class_counts[split][emotion] = count
            print(f"  {emotion}: {count} obrazów")
    print()

# ============================================================
# 3. WIZUALIZACJA ROZKŁADU KLAS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, split in enumerate(splits):
    if split in class_counts:
        emotions = list(class_counts[split].keys())
        counts = list(class_counts[split].values())
        axes[idx].bar(emotions, counts, color='skyblue', edgecolor='black')
        axes[idx].set_title(f"Rozkład klas - {split.upper()}")
        axes[idx].set_xlabel("Emocja")
        axes[idx].set_ylabel("Liczba obrazów")
        axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle("Analiza niezbalansowania klas FER-2013")
plt.tight_layout()
plt.show()

# ============================================================
# 4. PODGLĄD PRZYKŁADOWYCH OBRAZÓW
# ============================================================
from PIL import Image

fig, axes = plt.subplots(2, 7, figsize=(16, 6))
train_path = os.path.join(DATA_DIR, "train")

for idx, emotion in enumerate(sorted(os.listdir(train_path))):
    emotion_path = os.path.join(train_path, emotion)
    if os.path.isdir(emotion_path):
        images = os.listdir(emotion_path)
        if images:
            img = Image.open(os.path.join(emotion_path, images[0]))
            axes[0][idx].imshow(img, cmap='gray')
            axes[0][idx].set_title(emotion, fontsize=8)
            axes[0][idx].axis('off')

            img2 = Image.open(os.path.join(emotion_path, images[1]))
            axes[1][idx].imshow(img2, cmap='gray')
            axes[1][idx].axis('off')

plt.suptitle("Przykładowe obrazy per klasa emocji")
plt.tight_layout()
plt.show()

# ============================================================
# 5. DATA PIPELINE Z AUGMENTACJĄ
# ============================================================
print("=== Konfiguracja Data Pipeline z augmentacją ===\n")

# Augmentacja tylko dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalizacja [0,1]
    rotation_range=15,        # Rotacja ±15 stopni
    width_shift_range=0.1,    # Przesunięcie poziome
    height_shift_range=0.1,   # Przesunięcie pionowe
    horizontal_flip=True,     # Odbicie lustrzane
    zoom_range=0.1,           # Zoom
    brightness_range=[0.8, 1.2]  # Zmiana jasności
)

# Dla walidacji i testu TYLKO normalizacja (bez augmentacji)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_STATE
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nKlasy: {train_generator.class_indices}")
print(f"Liczba próbek treningowych: {train_generator.samples}")
print(f"Liczba próbek testowych: {test_generator.samples}")

# ============================================================
# 6. WIZUALIZACJA AUGMENTACJI
# ============================================================
print("\n=== Wizualizacja efektu augmentacji ===")

sample_emotion = sorted(os.listdir(train_path))[0]
sample_img_path = os.path.join(train_path, sample_emotion,
                               os.listdir(os.path.join(train_path, sample_emotion))[0])

sample_img = np.array(Image.open(sample_img_path).convert('L'))
sample_img = sample_img.reshape(1, 48, 48, 1).astype('float32') / 255.0

aug_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes[0][0].imshow(sample_img[0, :, :, 0], cmap='gray')
axes[0][0].set_title("Oryginał")
axes[0][0].axis('off')

aug_iter = aug_gen.flow(sample_img, batch_size=1)
for i in range(1, 10):
    aug_img = next(aug_iter)[0, :, :, 0]
    row, col = divmod(i, 5)
    axes[row][col].imshow(aug_img, cmap='gray')
    axes[row][col].set_title(f"Aug #{i}")
    axes[row][col].axis('off')

plt.suptitle(f"Efekt augmentacji - emocja: {sample_emotion}")
plt.tight_layout()
plt.show()

print("\n=== Zajęcia 2 zakończone! ===")
print(f"Train samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Klasy: {list(train_generator.class_indices.keys())}")