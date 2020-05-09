import numpy as np
import os
import nibabel as nib
import random
import time
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


def relu6(x):
    """Custom activation using relu6"""
    return K.relu(x, max_value=6)


def ConvBNrelu(x, filters=32, kernel=3, strides=1, padding="same"):
    """
    """
    x = layers.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    return x


def submodel(input_shape):
    """
    """
    inp = layers.Input(shape=input_shape + (1,))
    conv1 = ConvBNrelu(inp, filters=8, kernel=3, strides=1, padding="same")
    conv1 = ConvBNrelu(conv1, filters=8, kernel=3, strides=1, padding="same")
    conv1 = layers.MaxPooling2D()(conv1)

    conv2 = ConvBNrelu(conv1, filters=16, kernel=3, strides=1, padding="same")
    conv2 = ConvBNrelu(conv2, filters=16, kernel=3, strides=1, padding="same")
    conv2 = layers.MaxPooling2D()(conv2)

    conv3 = ConvBNrelu(conv2, filters=32, kernel=3, strides=1, padding="same")
    conv3 = ConvBNrelu(conv3, filters=32, kernel=3, strides=1, padding="same")
    conv3 = layers.MaxPooling2D()(conv3)

    out = layers.Flatten()(conv3)
    model = models.Model(inp, out)
    return model


def custom_model(input_shape=(32, 32), dropout=0.4, nclasses=None, multiencoders=True):
    """
    """

    inp1 = layers.Input(shape=input_shape + (1,), name="sagittal")
    inp2 = layers.Input(shape=input_shape + (1,), name="axial")
    inp3 = layers.Input(shape=input_shape + (1,), name="corronal")

    sagittal = submodel(input_shape)

    if multiencoders:
        axial = submodel(input_shape)
        corronal = submodel(input_shape)

        merge = [sagittal(inp1), axial(inp2), corronal(inp3)]
    else:
        merge = [sagittal(inp1), sagittal(inp2), sagittal(inp3)]

    concat = layers.Add()(merge)

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dropout(dropout)(out)

    out = layers.Dense(1, activation="sigmoid", name="output_node")(out)
    return models.Model(inputs=[inp1, inp2, inp3], outputs=out)
