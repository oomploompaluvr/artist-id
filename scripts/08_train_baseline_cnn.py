"""
Baseline CNN for Artist Identification
Person 2 - Deep Learning Implementation

Task:
Classify a painting into one of 15 artist categories.

Input:
128 x 128 RGB painting images

Output:
Predicted artist class
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    os.environ.get(
        "PAINTER_PROJECT_DIR",
        "/content/drive/MyDrive/PainterProject"
    )
)

IMAGE_DIR = PROJECT_DIR / "selected_images"
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 2. LOAD DATA AND LABEL MAPPING
# ============================================================

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

with open(DATA_DIR / "label_map.json", "r", encoding="utf-8") as file:
    label_map = json.load(file)

NUM_CLASSES = len(label_map)

print("Number of artists:", NUM_CLASSES)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def load_image(image_path, label):
    """Load, resize and normalize an RGB painting image."""

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    # Normalize pixel values from [0, 255] to [0, 1].
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(dataframe, training=False):
    """Create a TensorFlow dataset from a dataframe."""

    image_paths = [
        str(IMAGE_DIR / str(path))
        for path in dataframe["image_path"]
    ]

    labels = dataframe["label"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(dataframe), 2000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_dataset = create_dataset(train_df, training=True)
val_dataset = create_dataset(val_df)


# ============================================================
# 4. BUILD THE BASELINE CNN
# ============================================================

model = keras.Sequential(
    [
        layers.Input(shape=(128, 128, 3)),

        # Convolutional block 1
        layers.Conv2D(
            32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D(pool_size=2),

        # Convolutional block 2
        layers.Conv2D(
            64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D(pool_size=2),

        # Convolutional block 3
        layers.Conv2D(
            128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D(pool_size=2),

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ]
)


# ============================================================
# 5. COMPILE THE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# 6. TRAINING CALLBACKS
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "best_cnn.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 7. TRAIN THE MODEL
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# 8. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    RESULTS_DIR / "cnn_training_history.csv",
    index_label="epoch"
)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

best_model = keras.models.load_model(
    MODEL_DIR / "best_cnn.keras"
)

val_loss, val_accuracy = best_model.evaluate(
    val_dataset,
    verbose=1
)

print("\nBaseline CNN training complete!")
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

print(
    "Saved model:",
    MODEL_DIR / "best_cnn.keras"
)

print(
    "Saved training history:",
    RESULTS_DIR / "cnn_training_history.csv"
)