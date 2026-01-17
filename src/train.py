import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def train_model(model, base_model, train_gen, val_gen, class_weights):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.2),
        ModelCheckpoint("models/emotion_best.keras", save_best_only=True)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        class_weight=class_weights,
        callbacks=callbacks
    )

    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return model
