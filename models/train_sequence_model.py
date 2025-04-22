import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

# === CONFIGURAÇÕES ===
DATA_DIR = "../datasets/data-seq"
MODEL_PATH = "sequence_model.h5"
LABELS_PATH = "label_map.npy"
SEQUENCE_LENGTH = 20
FEATURES = 126  # 2 mãos

# === CARREGA AS SEQUÊNCIAS ===
sequences, labels = [], []

for label_dir in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label_dir)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(label_path, file))
            if sequence.shape == (SEQUENCE_LENGTH, FEATURES):
                sequences.append(sequence)
                labels.append(label_dir)

# === CONVERTE PARA ARRAYS ===
X = np.array(sequences)
y = np.array(labels)

# === ENCODING DOS RÓTULOS ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === DIVISÃO EM TREINO/TESTE ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# === DEFINIÇÃO DO MODELO LSTM ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TREINAMENTO ===
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# === SALVA MODELO E LABELS ===
Path("models").mkdir(exist_ok=True)
model.save(MODEL_PATH)
np.save(LABELS_PATH, label_encoder.classes_)

print(f"Modelo salvo em: {MODEL_PATH}")
print(f"Labels salvos em: {LABELS_PATH}")
