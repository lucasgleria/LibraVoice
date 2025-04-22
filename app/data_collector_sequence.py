import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# === CONFIGURAÇÕES ===
LABEL = "W"
NUM_SEQUENCES = 30
SEQUENCE_LENGTH = 20
FEATURES = 126  # 2 mãos × 21 × 3
SAVE_DIR = f"../datasets/data-seq/{LABEL}"

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(f"Gravando {NUM_SEQUENCES} sequências para gesto '{LABEL}'")

for seq_num in range(NUM_SEQUENCES):
    sequence = []
    print(f"→ Iniciando sequência {seq_num + 1}/{NUM_SEQUENCES}")
    frame_count = 0

    while frame_count < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmark_data = np.zeros(FEATURES)
        hand_data = []

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks[:2]:
                single_hand = []
                for lm in hl.landmark:
                    single_hand.extend([lm.x, lm.y, lm.z])
                hand_data.append(single_hand)

            if len(hand_data) == 1:
                landmark_data = np.array(hand_data[0] + [0]*63)
            elif len(hand_data) == 2:
                landmark_data = np.array(hand_data[0] + hand_data[1])

            sequence.append(landmark_data)
            frame_count += 1

            for hl in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Seq {seq_num+1} Frame {frame_count}/{SEQUENCE_LENGTH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Coletando sequência", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Salva sequência
    if len(sequence) == SEQUENCE_LENGTH:
        np.save(os.path.join(SAVE_DIR, f"{seq_num}.npy"), sequence)

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"Sequências salvas em {SAVE_DIR}")
