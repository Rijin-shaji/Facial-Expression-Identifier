import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_face(face, img_size):
    face = cv2.resize(face, (img_size, img_size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face.astype("float32"))
    return np.expand_dims(face, axis=0)
