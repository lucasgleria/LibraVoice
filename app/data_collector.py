import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from pathlib import Path

# === CONFIGURAÇÕES ===
LABEL = "S"  # Mude para o nome do gesto
NUM_SAMPLES = 200
FEATURES = 126  # 2 mãos × 21 landmarks × 3 coordenadas

# === CRIA DIRETÓRIO ===
Path("../datasets/data").mkdir(parents=True, exist_ok=True)
SAVE_PATH = f"../datasets/data/{LABEL}.csv"

# === CONFIGURA MEDIA PIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === INICIA CÂMERA ===
cap = cv2.VideoCapture(0)
collected = 0

print(f"Coletando dados para gesto: '{LABEL}'")

with open(SAVE_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)

    while collected < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmark_data = np.zeros(FEATURES)
        hand_data = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks[:2]:
                single_hand = []
                for lm in hand_landmarks.landmark:
                    single_hand.extend([lm.x, lm.y, lm.z])
                hand_data.append(single_hand)

            if len(hand_data) == 1:
                landmark_data = np.array(hand_data[0] + [0]*63)
            elif len(hand_data) == 2:
                landmark_data = np.array(hand_data[0] + hand_data[1])

            # Salva a linha com label
            row = landmark_data.tolist() + [LABEL]
            writer.writerow(row)
            collected += 1
            print(f"Amostra {collected}/{NUM_SAMPLES}")

            for hl in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Coletando dados estáticos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"Coleta finalizada em {SAVE_PATH}")
