import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# === CONFIGURAÇÕES ===
DATA_DIR = "../datasets/data"
MODEL_PATH = "model.pkl"
EXPECTED_FEATURES = 126

# === CARREGA TODOS OS CSVs ===
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado em datasets/data/.")

# Lê e concatena os dados
all_data = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    if df.shape[1] != EXPECTED_FEATURES + 1:
        print(f"Aviso: '{file}' tem {df.shape[1] - 1} features. Esperado: {EXPECTED_FEATURES}")
        continue
    all_data.append(df)

if not all_data:
    raise ValueError("Nenhum arquivo com 126 features encontrado.")

full_data = pd.concat(all_data, axis=0).reset_index(drop=True)

# === PREPARA DADOS ===
X = full_data.iloc[:, :-1].astype(float).values  # 126 features
y = full_data.iloc[:, -1].astype(str).values     # labels

# Divide em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TREINA MODELO ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Avalia
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acc:.2f}")

# Salva modelo
os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"Modelo salvo em: {MODEL_PATH}")
