import cv2

def load_face_detector(proto, weights):
    return cv2.dnn.readNetFromCaffe(proto, weights)
