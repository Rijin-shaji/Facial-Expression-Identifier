from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.config import IMG_SIZE, BATCH_SIZE

def create_generators(train_path, val_path, test_path):
    train_gen = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train = train_gen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val = test_gen.flow_from_directory(
        val_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test = test_gen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train, val, test
