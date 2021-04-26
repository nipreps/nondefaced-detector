"""Methods to predict using trained models"""

import os

import numpy as np
import tensorflow as tf

from pathlib import Path
from tqdm import tqdm

from nondefaced_detector.helpers import utils
from nondefaced_detector.models.model import CombinedClassifier


def _predict(volume, model, n_slices=32):
    """Return prediction from input volume.
    This is a helper function for the predict method.

    Parameters
    ----------
    volume: :obj:`np.ndarray`
        The nifti volume loaded as a numpy ndarray.
    model: :obj:`tf.keras.Model`
        The pretrained keras model loaded with weights.
    n_slices: int, optional, default=32
        The number of slices of the MRI volume to predict on.

    Returns
    -------
    float
        The predicted probability label from the sigmoid function.
    """

    if not isinstance(volume, (np.ndarray)):
        raise ValueError("volume is not a numpy ndarray")

    ds = _structural_slice(volume, plane="combined", n_slices=n_slices)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.batch(batch_size=1, drop_remainder=False)

    predicted = model.predict(ds)

    return predicted


def predict(volumes, model_path, n_slices=32):
    """Return predictions from a list of input volumes.

    Parameters
    ----------
    volumes: list
        A list of Path like strings to the volumes to make the
        prediction on.
    model_path: str - Path
        The path to pretrained model weights.
    n_slices: int, optional, default=32
        The number of 2D slices of the MRI volume to predict on.

    Returns
    -------
    list
        A list of predicted probabilities.
    """

    if not isinstance(volumes, list):
        raise ValueError(
            "Volumes need to be a list of paths to preprocessed MRI volumes."
        )

    outputs = []
    model = _get_model(model_path)

    for path in tqdm(volumes, total=len(volumes)):
        vol, _, _ = utils.load_vol(path)
        predicted = _predict(vol, model)

        outputs.append((path, predicted[0][0]))

    return outputs


def _structural_slice(x, plane, n_slices=16):
    """Transpose dataset and get slices from the volume based on
    the plane.

    Parameters
    ----------
    x: :obj:`tf.Tensor`
        The input MRI volume/dataset to sample slices from.
    plane: one of ["sagittal", "coronal", "axial", "combined"]
        The axes of the plane to get the slices for. If "combined", the
        input is sliced for all 3 axes.
    n_slices: int, optional, default=16
        The number of 2D slices to cut along the input plane. n_slices are
        randomly sampled from the input volume.

    Returns
    -------
    :obj:`tf.Tensor`
        A tensor of shape (n_slices, x.shape) or
        A dict with keys ['sagittal', 'coronal', 'axial'] each with a value
        of tensors of shape (n_slices, x.shape)
    """

    options = ["sagittal", "coronal", "axial", "combined"]

    if isinstance(plane, str) and plane in options:
        idxs = np.random.randint(x.shape[0], size=(n_slices, 3))
        if plane == "sagittal":
            midx = idxs[:, 0]
            x = x

        if plane == "coronal":
            midx = idxs[:, 1]
            x = tf.transpose(x, perm=[1, 2, 0])

        if plane == "axial":
            midx = idxs[:, 2]
            x = tf.transpose(x, perm=[2, 0, 1])

        if plane == "combined":
            temp = {}
            for op in options[:-1]:
                temp[op] = _structural_slice(x, op, n_slices)
            x = temp

        if not plane == "combined":
            x = tf.squeeze(tf.gather_nd(x, midx.reshape(n_slices, 1, 1)), axis=1)
            x = tf.math.reduce_mean(x, axis=0, keepdims=True)
            x = tf.expand_dims(x, axis=-1)
            x = tf.convert_to_tensor(x)

        return x
    else:
        raise ValueError(
            "Expected plane to be one of [sagittal, coronal, axial, combined]"
        )


def _get_model(model_path):
    """Return `tf.keras.Model` object from a filepath.

    Parameters
    ----------
    model_path: str, path to HDF5 or SavedModel file.

    Returns
    -------
    Instance of `tf.keras.Model`.

    Raises
    ------
    `ValueError` if cannot load model.
    """

    try:
        p = Path(model_path).resolve()

        model = CombinedClassifier(input_shape=(128, 128), wts_root=p, trainable=False)

        combined_weights = list(Path(os.path.join(p, "combined")).glob("*.h5"))[
            0
        ].resolve()

        model.load_weights(combined_weights)
        model.trainable = False

        return model

    except Exception as e:
        print(e)
        pass

    raise ValueError("Failed to load model.")
