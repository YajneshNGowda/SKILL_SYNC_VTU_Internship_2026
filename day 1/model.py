"""
model.py
────────
Model architectures for Plant Disease Detection.
Includes MobileNetV2 (primary), ResNet50 (comparison), and Custom CNN (baseline).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50


# ── Build Models ─────────────────────────────────────────────────────────────

def build_mobilenetv2(num_classes: int, img_size: tuple = (224, 224),
                      dropout: float = 0.4) -> tuple:
    """
    Build MobileNetV2 transfer learning model.

    Architecture:
      Input → MobileNetV2 (ImageNet, frozen) → GAP → Dense(256, ReLU)
      → Dropout → BN → Dense(num_classes, Softmax)

    Args:
        num_classes: Number of output classes.
        img_size:    Input image size (H, W).
        dropout:     Dropout rate in classification head.

    Returns:
        (model, base_model) — keep reference to base_model for fine-tuning.
    """
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    base.trainable = False    # Freeze during Phase 1

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='MobileNetV2_PlantDisease')
    return model, base


def build_resnet50(num_classes: int, img_size: tuple = (224, 224),
                   dropout: float = 0.5) -> tuple:
    """
    Build ResNet50 transfer learning model.
    Higher accuracy than MobileNetV2, but slower inference.

    Returns:
        (model, base_model)
    """
    base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='ResNet50_PlantDisease')
    return model, base


def build_custom_cnn(num_classes: int, img_size: tuple = (224, 224)) -> Model:
    """
    4-block Custom CNN from scratch — baseline model.
    Block structure: Conv2D → BatchNorm → MaxPool → Dropout

    Args:
        num_classes: Number of output classes.
        img_size:    Input image size (H, W).

    Returns:
        Keras Sequential model.
    """
    def conv_block(filters, kernel=3):
        return [
            layers.Conv2D(filters, (kernel, kernel), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25)
        ]

    model = tf.keras.Sequential([
        layers.Input(shape=(*img_size, 3)),
        *conv_block(32),
        *conv_block(64),
        *conv_block(128),
        *conv_block(256),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='Custom_CNN_Baseline')

    return model


# ── Training Pipeline ────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path: str, monitor: str = 'val_accuracy') -> list:
    """Standard set of training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    ]


