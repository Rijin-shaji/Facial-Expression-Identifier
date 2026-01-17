import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5

CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]
