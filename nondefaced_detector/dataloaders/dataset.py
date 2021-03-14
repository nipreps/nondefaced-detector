"""Method for creating tf.data.Dataset objects."""

import glob
import sys, os

import numpy as np
import tensorflow as tf

import nobrainer
from nobrainer.io import _is_gzipped
from nobrainer.volume import to_blocks

from nondefaced_detector.helpers.utils import load_vol

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(
    file_pattern,
    n_classes,
    batch_size,
    volume_shape,
    plane,
    n_slices=24,
    block_shape=None,
    n_epochs=None,
    mapping=None,
    shuffle_buffer_size=None,
    num_parallel_calls=AUTOTUNE,
    mode="train",
):

    """Returns tf.data.Dataset after preprocessing from
    tfrecords for training and validation

    Parameters
    ----------
    file_pattern:

    n_classes:
    """

    files = glob.glob(file_pattern)

    if not files:
        raise ValueError("no files found for pattern '{}'".format(file_pattern))

    compressed = _is_gzipped(files[0])
    shuffle = bool(shuffle_buffer_size)

    ds = nobrainer.dataset.tfrecord_dataset(
        file_pattern=file_pattern,
        volume_shape=volume_shape,
        shuffle=shuffle,
        scalar_label=True,
        compressed=compressed,
        num_parallel_calls=num_parallel_calls,
    )

    def _ss(x, y):

        x, y = structural_slice(x, y, plane, n_slices)
        return (x, y)

    ds = ds.map(_ss, num_parallel_calls)

    ds = ds.prefetch(buffer_size=batch_size)

    if batch_size is not None:
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if mode == "train":
        if shuffle_buffer_size:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat the dataset n_epochs times
        ds = ds.repeat(n_epochs)

    return ds


def structural_slice(x, y, plane, n_slices=4):

    """Transpose dataset based on the plane

    Parameters
    ----------
    x:

    y:

    plane:

    n:

    augment:
    """

    options = ["sagittal", "coronal", "axial", "combined"]
    shape = np.array(x.shape)
    if isinstance(plane, str) and plane in options:
        idxs = np.random.randint(x.shape[0], size=(n_slices, 3))
        #         idxs = np.array([[64, 64, 64]])
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
                temp[op] = structural_slice(x, y, op, n_slices)[0]
            x = temp

        if not plane == "combined":
            x = tf.squeeze(tf.gather_nd(x, midx.reshape(n_slices, 1, 1)), axis=1)
            x = tf.math.reduce_mean(x, axis=0)
            x = tf.expand_dims(x, axis=-1)
            x = tf.convert_to_tensor(x)
        return x, y
    else:
        raise ValueError("expected plane to be one of [sagittal, coronal, axial]")


if __name__ == "__main__":
    ROOTDIR = "/tf/shank/HDDLinux/Stanford/data/mriqc-shared/experiments/experiment_B/128/tfrecords_full"
    n_classes = 2
    global_batch_size = 8
    volume_shape = (128, 128, 128)
    ds = get_dataset(
        os.path.join(ROOTDIR, "data-train_*"),
        n_classes=n_classes,
        batch_size=global_batch_size,
        volume_shape=volume_shape,
        plane="coronal",
        shuffle_buffer_size=3,
    )

    import matplotlib.pyplot as plt

    # x, y = next(ds.as_numpy_operator())
    # print(x.shape, y)
    times = 0
    for x, y in ds.as_numpy_iterator():
        if times == 3:
            break
        print(x.shape, y)
        times += 1

        fig = plt.figure(figsize=(25, 8))

        for i in range(1, 9):
            fig.add_subplot(1, 8, i)
            plt.imshow(x[i - 1, :, :])

        print(ds)
