"""TensorFlow/Keras model definitions used in the original segmentation experiments."""

from __future__ import annotations

from typing import Any


def build_efficientnet(input_shape: tuple[int, int, int] = (600, 600, 3)) -> Any:
    """Build an EfficientNetB7 encoder with a transposed-convolution decoder."""
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB7

    backbone = EfficientNetB7(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    for filters in (256, 128, 64, 32, 16):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Cropping2D(((4, 4), (4, 4)))(x)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="efficientnet_b7_segmentation")


def build_deeplab(input_shape: tuple[int, int, int] = (600, 600, 3)) -> Any:
    """Build the ResNet50 + ASPP segmentation model explored by the project."""
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import ResNet50

    backbone = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False
    x = backbone.get_layer("conv4_block6_out").output

    branches = [layers.Conv2D(256, 1, padding="same", activation="relu")(x)]
    for rate in (6, 12, 18):
        branch = layers.Conv2D(256, 3, dilation_rate=rate, padding="same", use_bias=False)(x)
        branch = layers.BatchNormalization()(branch)
        branches.append(layers.ReLU()(branch))
    x = layers.Concatenate()(branches)
    x = layers.Conv2D(256, 1, padding="same", activation="relu")(x)

    for filters in (256, 128, 64, 32):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Cropping2D(((4, 4), (4, 4)))(x)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(backbone.input, outputs, name="resnet50_aspp_segmentation")
