import cv2
import numpy as np
import mediapipe as mp
import joblib
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from collections import deque

# Load ML model
model = joblib.load("focus_knn.pkl")

# MQTT setup
client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.connect("broker.emqx.io", 1883, 60)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Webcam
cap = cv2.VideoCapture(0)

ear_buffer = deque(maxlen=10)

def compute_ear(landmarks, w, h):
    left_eye = [33, 160, 158, 133, 153, 144]

    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in left_eye]

    vertical = np.linalg.norm(np.array(pts[1]) - np.array(pts[5])) + \
               np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))

    horizontal = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    ear = vertical / (2.0 * horizontal)
    return ear

print("Focus detector running... Press Q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        ear = compute_ear(landmarks, w, h)

        ear_buffer.append(ear)

        if len(ear_buffer) == 10:

            features = np.array(ear_buffer).reshape(1, -1)

            prediction = model.predict(features)[0]

            focus_value = float(prediction)

            client.publish("focus/status", focus_value)

            text = "Focused" if prediction == 1 else "Distracted"

            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)

            cv2.putText(frame, text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)
    else:
        ear_buffer.clear()

    cv2.imshow("Focus Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
