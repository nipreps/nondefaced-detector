"""Methods to predict using trained models"""

import os
import numpy as np
import tensorflow as tf

from pathlib import Path

from nondefaced_detector.models.modelN import CombinedClassifier
from nondefaced_detector.helpers import utils


def predict(
    input_volume,
    model_path,
    batch_size=1,
    n_samples=1,
    n_slices=32,
):
    """Return predictions from `inputs`.

    This is a general prediction method.

    Parameters
    ---------


    Returns
    ------
    """

    if n_samples < 1:
        raise Exception("n_samples cannot be lower than 1.")

    model = _get_model(model_path)

    ds = _structural_slice(input_volume, plane="combined", n_slices=n_slices)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)

    predicted = model.predict(ds)

    return predicted


def _structural_slice(x, plane, n_slices=16):

    """Transpose dataset based on the plane

    Parameters
    ----------
    x:

    plane:

    n_slices:

    Returns
    -------
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
    path: str, path to HDF5 or SavedModel file.

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


if __name__ == "__main__":

    from nondefaced_detector import preprocess
    from nondefaced_detector.helpers import utils

    wts_root = "models/pretrained_weights"

    vol_path = "../examples/sample_vols/IXI002-Guys-0828-T1.nii.gz"
    ppath, cpath = preprocess.preprocess(vol_path)

    volume, affine, _ = utils.load_vol(cpath)

    predicted = predict(volume, model_path=wts_root)

    print(predicted)
