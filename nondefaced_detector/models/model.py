"""Model definition for nondefaced-detector."""


import os
import tensorflow as tf

from tensorflow.keras import layers, models


def ConvBNrelu(x, filters=32, kernel=3, strides=1, padding="same"):
    """A layer block of one convolutional, one batch normalization,
    and one non-linear activation sequence.

    Parameters
    ----------
    x: :obj:`tf.Tensor` of rank 4+
        The input keras tensor object to instantiate a keras model
    filters: int, optional, default=32
        The dimensionality of the output space (i.e. the number of output
        filters in the convolution).
    kernel: int, optional, default=32
        An integer or tuple/list of 2 integers, specifying the height and width
        of the 2D convolution window. Can be a single integer to specify the same
        value for all spatial dimensions.
    strides: int
        Specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for all spatial dimensions.
    padding: one of "valid" or "same" (case-insensitive).
        "valid" means no padding. "same" results in padding evenly to the left/right
        or up/down of the input such that output has the same height/width dimension
        as the input.

    Returns
    -------
    :obj:`tf.Tensor`
        A tensor of rank 4+.
    """
    x = layers.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def TruncatedSubmodel(input_layer):
    """The TruncatedSubmodel trained in Step 1 of the model.

    Parameters
    ----------
    input_layer: tf.keras.Input
        The input keras tensor object to instantiate a keras model

    Returns
    -------
    :obj:`tf.Tensor`
        A flattened truncated network created from 3 sequential ConvBNRelu layer blocks
        joined by a MaxPooling layer.
    """
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
    """The final block of the model

    Parameters
    ----------
    layer: N-D tensor with shape: (batch_size, ..., input_dim)
        The flattened out feature layer output from the Submodels
    dropout: float
         Float between 0 and 1. Fraction of the input units to drop.

    Returns
    -------
    :obj:`tf.Tensor`
        N-D tensor with shape: (batch_size, ..., units)
    """
    out = layers.Dense(256, activation="relu")(layer)
    out = layers.Dropout(dropout)(out)
    out = layers.Dense(1, activation="sigmoid", name="output_node")(out)
    return out


def Submodel(
    root_path,
    input_shape=(32, 32),
    dropout=0.4,
    name="axial",
    weights="axial",
    include_top=True,
    trainable=True,
):
    """3 identical submodel blocks are used to train on spatial information
    from all three axes (axial, coronal, sagittal) separately.

    Parameters
    ----------
    root_path: str, Path
        Root directory for storing the weights.
    input_shape: tuple of ints, default=(32, 32)
        The shape of the input image.
    dropout: float, optional, default=0.4
        Float between 0 and 1. Fraction of the input units to drop.
    name: str
        Name of the submodel.
    weights: str
        Name of the folder to store the weights for the submodel.
    include_top: bool, default=True
        If True, the the model includes the ClassiferHead block at the
        end.
    trainable: bool, default=True
        If True, the model is set to be trainable else the model layers
        are frozen.

    Returns
    -------
    `tf.keras.Model`
        Returns a `tf.keras.Model` object with features.
    """

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
    """The final block of the model that combines features and outputs a real-valued
    probability using the sigmoid function.

    Parameters
    ----------
    input_shape: tuple of ints, default=(32, 32)
        The shape of the input image.
    dropout: float, optional, default=0.4
        Float between 0 and 1. Fraction of the input units to drop.
    trainable: bool, default=True
        If True, the model is set to be trainable else the model layers
        are frozen.
    shared: bool, default=False

    Returns
    -------

    """

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

        merge_features = [axial_features(p1), axial_features(p2), axial_features(p3)]
        input_features = [p1, p2, p3]

    add_features = layers.Add()(merge_features)
    prob = ClassifierHead(add_features, dropout)

    model = models.Model(inputs=input_features, outputs=prob)

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
