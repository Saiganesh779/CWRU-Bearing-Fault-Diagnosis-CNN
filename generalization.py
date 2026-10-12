import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "CWRU_48k_load_1_CNN_data.npz"

EPOCHS = 30
BATCH_SIZE = 32

# 80% of each class -> training
# 20% of each class -> unseen test
TRAIN_RATIO = 0.80

RANDOM_SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING CWRU BEARING DATASET")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{os.path.abspath(DATASET_PATH)}"
        )

    data = np.load(
        DATASET_PATH,
        allow_pickle=True
    )

    X = data["data"]
    labels = data["labels"]

    print("\nOriginal data shape:", X.shape)
    print("Labels shape:", labels.shape)

    # Convert to float32
    X = X.astype("float32")

    # Add channel dimension for CNN
    # (4600, 32, 32) -> (4600, 32, 32, 1)
    X = X[..., np.newaxis]

    # Encode string labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    print("\nNumber of classes:", len(encoder.classes_))

    print("\nClasses:")
    for i, class_name in enumerate(encoder.classes_):
        count = np.sum(y == i)
        print(f"  {i}: {class_name} -> {count} samples")

    return X, y, encoder


# ============================================================
# CREATE ORDERED HOLDOUT SPLIT
# ============================================================

def create_ordered_split(X, y):

    print("\n" + "=" * 70)
    print("CREATING GENERALIZATION TEST SPLIT")
    print("=" * 70)

    print(
        "\nUsing an ordered-within-class split:"
        "\n  First 80% of each class -> training"
        "\n  Last 20% of each class  -> unseen test"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is only a proxy for generalization."
        "\nThe NPZ file does not contain original recording IDs."
    )

    train_indices = []
    test_indices = []

    classes = np.unique(y)

    for class_id in classes:

        class_indices = np.where(y == class_id)[0]

        split_point = int(
            len(class_indices) * TRAIN_RATIO
        )

        train_class_indices = class_indices[:split_point]
        test_class_indices = class_indices[split_point:]

        train_indices.extend(
            train_class_indices
        )

        test_indices.extend(
            test_class_indices
        )

    train_indices = np.array(
        train_indices,
        dtype=int
    )

    test_indices = np.array(
        test_indices,
        dtype=int
    )

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    print("\nTraining samples:", len(X_train))
    print("Test samples:", len(X_test))

    print("\nTraining samples per class:")
    for class_id in classes:
        print(
            f"  {class_id}: "
            f"{np.sum(y_train == class_id)}"
        )

    print("\nTest samples per class:")
    for class_id in classes:
        print(
            f"  {class_id}: "
            f"{np.sum(y_test == class_id)}"
        )

    return X_train, y_train, X_test, y_test


# ============================================================
# BUILD CNN
# ============================================================

def build_model(num_classes):

    print("\n" + "=" * 70)
    print("BUILDING CNN")
    print("=" * 70)

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(32, 32, 1)
        ),

        # Block 1
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 2
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 3
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(),

        # Fully connected
        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        # Output
        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model.summary()

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train):

    print("\n" + "=" * 70)
    print("TRAINING CNN")
    print("=" * 70)

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    history = model.fit(

        X_train,
        y_train,

        validation_split=0.15,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1,

        shuffle=True
    )

    return history


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    encoder
):

    print("\n" + "=" * 70)
    print("GENERALIZATION TEST RESULTS")
    print("=" * 70)

    # Predict
    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # F1
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print("\n-----------------------------------")
    print("OVERALL RESULTS")
    print("-----------------------------------")

    print(
        f"Test accuracy : {accuracy:.4f}"
    )

    print(
        f"Test accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1      : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1   : {weighted_f1:.4f}"
    )

    # Classification report
    print("\n-----------------------------------")
    print("CLASSIFICATION REPORT")
    print("-----------------------------------")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            digits=4
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n-----------------------------------")
    print("CONFUSION MATRIX")
    print("-----------------------------------")

    print(cm)

    # Per-class accuracy
    print("\n-----------------------------------")
    print("PER-CLASS ACCURACY")
    print("-----------------------------------")

    for i, class_name in enumerate(
        encoder.classes_
    ):

        total = np.sum(
            y_test == i
        )

        correct = cm[i, i]

        class_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        print(
            f"{class_name:10s}: "
            f"{class_accuracy * 100:.2f}% "
            f"({correct}/{total})"
        )

    return accuracy, macro_f1, weighted_f1


# ============================================================
# CHECK EXACT DUPLICATES
# ============================================================

def check_duplicates(X):

    print("\n" + "=" * 70)
    print("DUPLICATE SAMPLE CHECK")
    print("=" * 70)

    # Remove channel dimension
    X_flat = X.reshape(
        X.shape[0],
        -1
    )

    # Number of unique samples
    unique_count = np.unique(
        X_flat,
        axis=0
    ).shape[0]

    duplicate_count = (
        X.shape[0] - unique_count
    )

    print(
        "\nTotal samples:",
        X.shape[0]
    )

    print(
        "Unique samples:",
        unique_count
    )

    print(
        "Exact duplicate samples:",
        duplicate_count
    )

    if duplicate_count == 0:

        print(
            "\nNo exact duplicate samples "
            "were found."
        )

    else:

        print(
            "\nWARNING: Exact duplicate "
            "samples exist."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CWRU BEARING FAULT CNN")
    print("GENERALIZATION TEST")
    print("=" * 70)

    # Reproducibility
    np.random.seed(
        RANDOM_SEED
    )

    tf.random.set_seed(
        RANDOM_SEED
    )

    # Load
    X, y, encoder = load_dataset()

    # Check duplicates
    check_duplicates(X)

    # Split
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = create_ordered_split(
        X,
        y
    )

    # Build
    model = build_model(
        len(encoder.classes_)
    )

    # Train
    train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate
    accuracy, macro_f1, weighted_f1 = evaluate_model(
        model,
        X_test,
        y_test,
        encoder
    )

    # Save model
    output_model = (
        "bearing_fault_generalization.keras"
    )

    model.save(
        output_model
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"\nSaved to:\n{os.path.abspath(output_model)}"
    )

    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        "\nThis test uses an ordered holdout because "
        "the NPZ file does not contain recording IDs."
    )

    print(
        "\nTherefore, this result can be used as a "
        "generalization check, but it CANNOT prove "
        "recording-independent generalization."
    )

    print(
        "\nFor a definitive test, the original CWRU "
        "recording/source ID must be available for "
        "each vibration window."
    )

    print("\nDONE.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()