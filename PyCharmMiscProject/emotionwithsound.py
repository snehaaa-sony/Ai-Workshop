import cv2
from deepface import DeepFace
import pyttsx3
import threading

# Initialize TTS engine
engine = pyttsx3.init()

# Function to speak emotion (in a separate thread to avoid GUI lag)
def speak_emotion(emotion):
    threading.Thread(target=lambda: engine.say(f" {emotion}"), daemon=True).start()
    threading.Thread(target=engine.runAndWait, daemon=True).start()

cap = cv2.VideoCapture(0)

last_emotion = None  # Track last emotion to avoid repeating the same emotion

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = results[0]['dominant_emotion']

        if emotion != last_emotion:
            speak_emotion(emotion)
            last_emotion = emotion

        cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    except Exception as e:
        cv2.putText(frame, f'Error: {str(e)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()