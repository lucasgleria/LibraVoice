import pyttsx3

# Inicializa uma vez só
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # velocidade da fala
engine.setProperty('volume', 1.0)

def speak_text(text):
    if not text.strip():
        return
    engine.say(text)
    engine.runAndWait()
