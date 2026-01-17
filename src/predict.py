import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from src.config import IMG_SIZE, EMOTIONS, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW
from src.inference.face_detector import load_face_detector
from src.inference.emotion_predictor import preprocess_face

model = load_model("models/iris_cnn.h5")
face_net = load_face_detector(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

history = deque(maxlen=SMOOTHING_WINDOW)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            inp = preprocess_face(face, IMG_SIZE)
            preds = model.predict(inp, verbose=0)[0]

            history.append(preds)
            avg_preds = np.mean(history, axis=0)

            idx = np.argmax(avg_preds)
            conf = avg_preds[idx]
            emotion = EMOTIONS[idx] if conf >= CONFIDENCE_THRESHOLD else "Neutral"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} {conf*100:.1f}%",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
