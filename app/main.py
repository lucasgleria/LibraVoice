
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
from collections import deque
import os
from tts import speak_text
import time


# Configurações
SEQUENCE_LENGTH = 20
FEATURES = 126
MOVEMENT_THRESHOLD = 0.08  # Limite para decidir se a mão está em movimento

# Caminhos dos modelos
STATIC_MODEL_PATH = "../models/model.pkl"
SEQUENCE_MODEL_PATH = "../models/sequence_model.h5"
LABELS_PATH = "../models/label_map.npy"

# Carrega os modelos
static_model = joblib.load(STATIC_MODEL_PATH)
sequence_model = tf.keras.models.load_model(SEQUENCE_MODEL_PATH)
labels = np.load(LABELS_PATH)

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializa buffer
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
last_static_landmark = None

# Inicia câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
print("Pressione 'q' para sair.")

# TTS
texto_final = ""
last_prediction = ""

# Delay
last_spoken_time = 0
DELAY_ENTRE_AUDIO = 1.5  # segundos


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction_text = ""
    landmark_data = np.zeros(FEATURES)  # zera vetor de 126 por padrão
    hand_data = []  # guarda landmarks separados
    
    # if not result.multi_hand_landmarks:
    #     continue  # nada a processar, pula para próximo frame   
    
    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks[:2]):
            single_hand = []
            for lm in hand_landmarks.landmark:
                single_hand.extend([lm.x, lm.y, lm.z])
            hand_data.append(single_hand)

            # Desenha a mão na tela
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Monta vetor de entrada (duas mãos)
        if len(hand_data) == 1:
            # Uma mão → preenche a outra com zeros
            landmark_data = np.array(hand_data[0] + [0]*63)
        else:
            landmark_data = np.array(hand_data[0] + hand_data[1])

        # Armazena para sequência
        sequence_buffer.append(landmark_data)

        # Detecta movimento (compara com última amostra)
        movement = 0
        if last_static_landmark is not None:
            movement = np.mean(np.abs(landmark_data - last_static_landmark))
        last_static_landmark = landmark_data

        # Decide modo de predição
        if len(sequence_buffer) == SEQUENCE_LENGTH and movement > MOVEMENT_THRESHOLD:
            input_seq = np.expand_dims(sequence_buffer, axis=0)
            pred = sequence_model.predict(input_seq)[0]
            predicted_label = labels[np.argmax(pred)]
            confidence = np.max(pred)
            prediction_text = f"[M] {predicted_label} ({confidence:.2f})"
        elif movement <= MOVEMENT_THRESHOLD:
            input_static = landmark_data.reshape(1, -1)
            pred = static_model.predict(input_static)[0]
            predicted_label = pred
            prediction_text = f"[S] {pred}"

    # Mostra na tela
    if prediction_text:
        cv2.putText(frame, prediction_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Detecta mudança no gesto e fala
        current_gesture = predicted_label
        current_time = time.time()

        if current_gesture and current_gesture != last_prediction:
            if current_time - last_spoken_time > DELAY_ENTRE_AUDIO:
                texto_final += current_gesture + " "
                speak_text(current_gesture)
                last_prediction = current_gesture
                last_spoken_time = current_time



    cv2.imshow("Reconhecimento Automático", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
hands.close()