import numpy as np
import os
import nibabel as nib
import random
import time
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


def ConvBNrelu(x, filters=32, kernel=3, strides=1, padding="same"):
    """"""
    x = layers.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def TruncatedSubmodel(input_layer):
    """"""
    conv1 = ConvBNrelu(input_layer, filters=8, kernel=3, strides=1, padding="same")
    conv1 = ConvBNrelu(conv1, filters=8, kernel=3, strides=1, padding="same")
    conv1 = layers.MaxPooling2D()(conv1)

    conv2 = ConvBNrelu(conv1, filters=16, kernel=3, strides=1, padding="same")
    conv2 = ConvBNrelu(conv2, filters=16, kernel=3, strides=1, padding="same")
    conv2 = layers.MaxPooling2D()(conv2)

    conv3 = ConvBNrelu(conv2, filters=32, kernel=3, strides=1, padding="same")
    conv3 = ConvBNrelu(conv3, filters=32, kernel=3, strides=1, padding="same")
    conv3 = layers.MaxPooling2D()(conv3)

    out = layers.Flatten()(conv3)
    return out


def ClassifierHead(layer, dropout):
    """"""
    out = layers.Dense(256, activation="relu")(layer)
    out = layers.Dropout(dropout)(out)
    out = layers.Dense(1, activation="sigmoid", name="output_node")(out)
    return out


def Submodel(
    input_shape=(32, 32),
    dropout=0.4,
    name="axial",
    weights="axial",
    include_top=True,
    root_path=None,
    trainable=True,
):
    """"""
    input_layer = layers.Input(shape=input_shape + (1,), name=name)
    features = TruncatedSubmodel(input_layer)

    if not include_top:
        model = models.Model(input_layer, features)
    else:
        classifier = ClassifierHead(features, dropout)
        model = models.Model(input_layer, classifier)

    if weights:
        weights_pth = os.path.join(root_path, name, "best-wts.h5")
        model.load_weights(weights_pth)

    if not trainable:
        for layer in model.layers:
            layer.trainable = False

    return model


def CombinedClassifier(
    input_shape=(32, 32), dropout=0.4, wts_root=None, trainable=False, shared=False
):
    """"""

    axial_features = Submodel(
        input_shape,
        dropout,
        name="axial",
        weights=None,
        include_top=False,
        root_path=wts_root,
    )

    if not shared:
        sagittal_features = Submodel(
            input_shape,
            dropout,
            name="sagittal",
            weights=None,
            include_top=False,
            root_path=wts_root,
        )
        coronal_features = Submodel(
            input_shape,
            dropout,
            name="coronal",
            weights=None,
            include_top=False,
            root_path=wts_root,
        )

        input_features = [
            axial_features.inputs,
            coronal_features.inputs,
            sagittal_features.inputs,
        ]

        merge_features = [
            axial_features.outputs[0],
            sagittal_features.outputs[0],
            coronal_features.outputs[0],
        ]

    else:

        p1 = layers.Input(shape=input_shape + (1,), name="plane1")
        p2 = layers.Input(shape=input_shape + (1,), name="plane2")
        p3 = layers.Input(shape=input_shape + (1,), name="plane3")

        merge_features = [
            axial_features(p1),
            axial_features(p2),
            axial_features(p3),
        ]
        input_features = [p1, p2, p3]

    add_features = layers.Add()(merge_features)
    prob = ClassifierHead(add_features, dropout)

    model = models.Model(
        inputs=input_features,
        outputs=prob,
    )

    if not trainable:
        assert not (wts_root == None)
        axial_model = Submodel(
            input_shape,
            dropout,
            name="axial",
            weights="axial",
            include_top=True,
            root_path=wts_root,
        )
        coronal_model = Submodel(
            input_shape,
            dropout,
            name="coronal",
            weights="coronal",
            include_top=True,
            root_path=wts_root,
        )
        sagittal_model = Submodel(
            input_shape,
            dropout,
            name="sagittal",
            weights="sagittal",
            include_top=True,
            root_path=wts_root,
        )

        for ii in range(1, len(axial_features.layers)):
            model.layers[3 * ii].set_weights(axial_model.layers[ii].get_weights())
            model.layers[3 * ii + 1].set_weights(
                sagittal_model.layers[ii].get_weights()
            )
            model.layers[3 * ii + 2].set_weights(coronal_model.layers[ii].get_weights())

            model.layers[3 * ii].trainable = False
            model.layers[3 * ii + 1].trainable = False
            model.layers[3 * ii + 1].trainable = False

    return model


"""
if __name__ == '__main__':
    axial = Submodel(name='axial', weights=None)
    coronal = Submodel(name='coronal', weights=None)
    sagittal = Submodel(name='sagittal', weights = None)
    
    combined = CombinedClassifier()
    print (combined.summary())

"""
