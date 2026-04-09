import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# =========================
# 1. DATASET PATH
# =========================
DATASET_PATH = r"C:\Users\HP\Desktop\speech project\dataset\patient-vocal-dataset"

# =========================
# 2. CLASS LABELS (MATCH EXACT FOLDERS)
# =========================
classes = ["Healthy", "Laryngozele", "Vox sensils"]

# =========================
# 3. DATA STORAGE
# =========================
X = []
y = []

# =========================
# 4. FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to 128x128
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)

    if mel_db.shape[0] < 128:
        mel_db = np.pad(mel_db, ((0, 128 - mel_db.shape[0]), (0, 0)))

    mel_db = mel_db[:128, :128]

    return mel_db

# =========================
# 5. LOAD DATASET
# =========================
for label, class_name in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    print(f"📂 Loading: {class_name}")

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

# =========================
# 6. CHECK DATA
# =========================
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("❌ No data loaded. Check dataset path and folder names.")

print("✅ Dataset loaded:", X.shape)

# =========================
# 7. PREPROCESS
# =========================
X = X.reshape(X.shape[0], 128, 128, 1)
y = to_categorical(y, num_classes=len(classes))

# =========================
# 8. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 9. MODEL
# =========================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# =========================
# 10. COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 11. TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)

# =========================
# 12. SAVE MODEL
# =========================
model.save("saved_model.keras")

print("✅ Training completed successfully!")