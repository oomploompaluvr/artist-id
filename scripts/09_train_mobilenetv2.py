"""
MobileNetV2 Transfer Learning
Artist Identification - 15 Artists

Uses an ImageNet-pretrained MobileNetV2 backbone
and a custom classification head.
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
EPOCHS = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

with open(
    DATA_DIR / "label_map.json",
    "r",
    encoding="utf-8"
) as file:
    label_map = json.load(file)

NUM_CLASSES = len(label_map)

print("Number of artists:", NUM_CLASSES)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def load_image(image_path, label):
    """Load an image and normalize its pixel values."""

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    # Normalize pixel values to [0, 1].
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(dataframe, training=False):
    """Build an efficient TensorFlow dataset."""

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
# 4. LOAD PRETRAINED MOBILENETV2
# ============================================================

base_model = keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained convolutional layers.
base_model.trainable = False


# ============================================================
# 5. BUILD THE CLASSIFICATION MODEL
# ============================================================

inputs = keras.Input(shape=(128, 128, 3))

# Convert [0, 1] pixel values to [-1, 1].
x = layers.Rescaling(
    scale=2.0,
    offset=-1.0
)(inputs)

# Extract visual features using MobileNetV2.
x = base_model(x, training=False)

# Classification head.
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="artist_mobilenetv2"
)

model.summary()


# ============================================================
# 6. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 7. CALLBACKS
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "best_mobilenet.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

training_logger = keras.callbacks.CSVLogger(
    str(RESULTS_DIR / "mobilenet_training_history.csv"),
    append=False
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        training_logger
    ]
)


# ============================================================
# 9. LOAD BEST CHECKPOINT
# ============================================================

best_model = keras.models.load_model(
    MODEL_DIR / "best_mobilenet.keras"
)

val_loss, val_accuracy = best_model.evaluate(
    val_dataset,
    verbose=1
)


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    RESULTS_DIR / "mobilenet_training_history.csv",
    index_label="epoch"
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\nMobileNetV2 training complete!")

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

print(
    "Best model saved at:",
    MODEL_DIR / "best_mobilenet.keras"
)

print(
    "Training history saved at:",
    RESULTS_DIR / "mobilenet_training_history.csv"
)