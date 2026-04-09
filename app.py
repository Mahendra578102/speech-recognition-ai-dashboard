from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import librosa.display
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("saved_model.keras")

classes = ["Healthy", "Laryngozele", "Vox sensils"]

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)

    if mel_db.shape[0] < 128:
        mel_db = np.pad(mel_db, ((0, 128 - mel_db.shape[0]), (0, 0)))

    return mel_db[:128, :128]


# Spectrogram generation (FIXED)
def generate_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    plt.figure(figsize=(5, 4))

    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

    path = os.path.join(STATIC_FOLDER, "spectrogram.png")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return "spectrogram.png"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Generate spectrogram
    spectrogram_file = generate_spectrogram(filepath)

    # Prediction
    features = extract_features(filepath)
    features = features.reshape(1, 128, 128, 1)

    prediction = model.predict(features)
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "spectrogram": spectrogram_file
    })


# Accuracy graph data
@app.route("/accuracy")
def accuracy():
    with open("history.json", "r") as f:
        history = json.load(f)

    return jsonify({
        "accuracy": history["accuracy"],
        "val_accuracy": history["val_accuracy"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)