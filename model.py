"""
Bearing fault detection - CNN classifier for CWRU_48k_load_1_CNN_data.npz

This dataset is already preprocessed:
    data   : (4600, 32, 32) float64  - vibration windows reshaped to 32x32 "images"
    labels : (4600,) string          - one of 10 classes:
                 Normal
                 IR_007, IR_014, IR_021   (inner race fault, 3 severities)
                 OR_007, OR_014, OR_021   (outer race fault, 3 severities)
                 Ball_007, Ball_014, Ball_021  (ball fault, 3 severities)
Each class has 460 samples - perfectly balanced, so plain accuracy is a fair metric.

Pipeline:
1. Load the .npz
2. Encode string labels -> integers, reshape data to (N, 32, 32, 1) for Conv2D
3. Stratified train/val/test split
4. Train a small CNN
5. Evaluate on the held-out test set with a classification report + confusion matrix
6. Save the trained model and the label encoder

Usage:
    python cwru_cnn_train.py
        (looks for CWRU_48k_load_1_CNN_data.npz in the current folder)
    python cwru_cnn_train.py /path/to/some_other_file.npz
        (uses that file instead)
"""

import os
import sys

DEFAULT_NPZ_FILENAME = "CWRU_48k_load_1_CNN_data.npz"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["data"].astype("float32")
    y_raw = d["labels"]

    # Add channel dimension: (N, 32, 32) -> (N, 32, 32, 1)
    X = X[..., np.newaxis]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, encoder


def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(npz_path):
    X, y, encoder = load_data(npz_path)
    num_classes = len(encoder.classes_)
    print(f"Loaded {X.shape[0]} samples, shape {X.shape[1:]}, {num_classes} classes:")
    print(" ", list(encoder.classes_))

    # 70% train / 15% val / 15% test, stratified so each split keeps class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save("bearing_fault_cnn.keras")
    joblib.dump(encoder, "bearing_fault_label_encoder.joblib")
    print("\nSaved model to bearing_fault_cnn.keras")
    print("Saved label encoder to bearing_fault_label_encoder.joblib")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        npz_path = sys.argv[1]
    elif len(sys.argv) == 1:
        # No path given - look for the default filename next to this script.
        npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_NPZ_FILENAME)
        if not os.path.exists(npz_path):
            print(f"Could not find '{DEFAULT_NPZ_FILENAME}' next to this script.")
            print("Either place the .npz file in the same folder as this script, or run:")
            print("    python cwru_cnn_train.py /path/to/your_file.npz")
            sys.exit(1)
    else:
        print("Usage: python cwru_cnn_train.py [/path/to/file.npz]")
        sys.exit(1)

    main(npz_path)