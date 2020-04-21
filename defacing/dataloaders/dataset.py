import nobrainer
from nobrainer.io import _is_gzipped
from nobrainer.volume import to_blocks

import tensorflow as tf
import glob
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
ROOTDIR = '/work/06850/sbansal6/maverick2/mriqc-shared/'

# function to apply augmentations to tf dataset
def apply_augmentations(features, labels):

    """ Apply <TYPE_OF> augmentation to the dataset
    
    """
    #     iaa.SomeOf(
    #             (0, 3),
    #             [
    #                 iaa.Fliplr(0.5),
    #                 iaa.Flipud(0.5),
    #                 iaa.Noop(),
    #                 iaa.OneOf(
    #                     [
    #                         iaa.Affine(rotate=90),
    #                         iaa.Affine(rotate=180),
    #                         iaa.Affine(rotate=270),
    #                     ]
    #                 ),
    #                 # iaa.GaussianBlur(sigma=(0.0, 0.2)),
    #             ],
    #         )

    return


# from nobrainer.dataset
def tfrecord_dataset(
    file_pattern,
    volume_shape,
    shuffle,
    scalar_label,
    compressed=True,
    num_parallel_calls=None,
):
    """Return `tf.data.Dataset` from TFRecord files."""

    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    # Read each of these files as a TFRecordDataset.
    # Assume all files have same compression type as the first file.
    compression_type = "GZIP" if compressed else None
    cycle_length = 1 if num_parallel_calls is None else num_parallel_calls
    dataset = dataset.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(
            x, compression_type=compression_type
        ),
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
    )
    parse_fn = parse_example_fn(volume_shape=volume_shape, scalar_label=scalar_label)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
    return dataset


# from nobrainer.tfrecord
def parse_example_fn(volume_shape, scalar_label=False):
    """Return function that can be used to read TFRecord file into tensors.
    Parameters
    ----------
    volume_shape: sequence, the shape of the feature data. If `scalar_label` is `False`,
        this also corresponds to the shape of the label data.
    scalar_label: boolean, if `True`, label is a scalar. If `False`, label must be the
        same shape as feature data.
    Returns
    -------
    Function with which a TFRecord file can be parsed.
    """

    @tf.function
    def parse_example(serialized):
        """Parse one example from a TFRecord file made with Nobrainer.
        Parameters
        ----------
        serialized: str, serialized proto message.
        Returns
        -------
        Tuple of two tensors. If `scalar_label` is `False`, both tensors have shape
        `volume_shape`. Otherwise, the first tensor has shape `volume_shape`, and the
        second is a scalar tensor.
        """
        features = {
            "feature/shape": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "feature/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "label/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "label/rank": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        }
        e = tf.io.parse_single_example(serialized=serialized, features=features)
        x = tf.io.decode_raw(e["feature/value"], _TFRECORDS_DTYPE)
        y = tf.io.decode_raw(e["label/value"], _TFRECORDS_DTYPE)
        # TODO: this line does not work. The shape cannot be determined
        # dynamically... for now.
        # xshape = tf.cast(
        #     tf.io.decode_raw(e["feature/shape"], _TFRECORDS_DTYPE), tf.int32)
        x = tf.reshape(x, shape=volume_shape)
        if not scalar_label:
            y = tf.reshape(y, shape=volume_shape)
        else:
            y = tf.reshape(y, shape=[1])
        return x, y

    return parse_example


def standardize(x, clip_value_min=-5000, clip_value_max=5000):
    """Standard score input tensor.
    Implements `(x - mean(x)) / stdev(x)`.
    Parameters
    ----------
    x: tensor, values to standardize.
    Returns
    -------
    Tensor of standardized values. Output has mean 0 and standard deviation 1.
    """
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
    x = tf.convert_to_tensor(x)
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    # x = tf.clip_by_value(
    #     x, 
    #    clip_value_min, 
    #    clip_value_max, name=None
    #    )
    
    mean, var = tf.nn.moments(x, axes=None)
    std = tf.sqrt(var)
    return (x - mean) / std


def normalize(x):
    """Standard score input tensor.
    Implements `(x - mean(x)) / stdev(x)`.
    Parameters
    ----------
    x: tensor, values to standardize.
    Returns
    -------
    Tensor of standardized values. Output has mean 0 and standard deviation 1.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)

    max_value = tf.math.reduce_max(
                x, 
                axis=None, 
                keepdims=False, name=None
                )

    min_value = tf.math.reduce_min(
                x, 
                axis=None, 
                keepdims=False, name=None
                )
    return (x - min_value) / (max_value - min_value + 1e-3)

def get_dataset(
    file_pattern,
    n_classes,
    batch_size,
    volume_shape,
    plane,
    n = 4,
    block_shape=None,
    n_epochs=None,
    mapping=None,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=AUTOTUNE,
):

    """ Returns tf.data.Dataset after preprocessing from 
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

    if augment:
        ds = ds.map(
            lambda x, y: tf.cond(
                tf.random.uniform((1,)) > 0.5,
                true_fn=lambda: apply_augmentations(x, y),
                false_fn=lambda: (x, y),
            ),
            num_parallel_calls=num_parallel_calls,
        )

    def _ss(x, y):
        x, y = structural_slice(x, y, plane, n)
        return (x, y)

    ds = ds.map(_ss, num_parallel_calls)

    #     def _f(x, y):
    #         x = to_blocks(x, block_shape)
    #         n_blocks = x.shape[0]
    #         y = tf.repeat(y, n_blocks)
    #         return (x, y)
    #     ds = ds.map(_f, num_parallel_calls=num_parallel_calls)

    # This step is necessary because it reduces the extra dimension.
    # ds = ds.unbatch()

    # add a single dimension at the end
    # ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y))

    ds = ds.prefetch(buffer_size=batch_size)

    if batch_size is not None:
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        ds = ds.map(lambda x,y: (tf.reshape(x, ((batch_size*n, 3, ) if plane == "combined" else (batch_size*n,)) + volume_shape[:2] +(1,)), 
                                tf.reshape(y, (batch_size*n,))))

    if shuffle_buffer_size:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat the dataset n_epochs times
    ds = ds.repeat(n_epochs)

    return ds


def structural_slice(x, y, plane, n=4):

    """ Transpose dataset based on the plane
    
    Parameters
    ----------
    x:
    
    y:
    
    plane:
    
    """

    options = ["axial", "coronal", "sagittal", "combined"]
    shape = np.array(x.shape)
    x = normalize(standardize(x))

    if isinstance(plane, str) and plane in options:
        if plane == "axial":
            idx = np.random.randint(shape[0]//5, 3*shape[0]//5, n)
            x = x

        if plane == "coronal":
            idx = np.random.randint(shape[1]//4, 3*shape[1]//4, n)
            x = tf.transpose(x, perm=[1, 2, 0])

        if plane == "sagittal":
            idx = np.random.randint(shape[2]//3, 2*shape[2]//3, n)
            x = tf.transpose(x, perm=[2, 0, 1])

        if plane == "combined":
            temp = []
            for op in options[:-1]:
                temp.append(tf.expand_dims(structural_slice(x, y, op, n)[0], axis=1))
            x = tf.concat(temp, axis = 1)

        if not plane == "combined": x = tf.squeeze(tf.gather_nd(x, idx.reshape(n, 1, 1)), axis=1)
        x = tf.convert_to_tensor(x)
        y = tf.repeat(y, n)
        
        return x, y
    else:
        raise ValueError("expected plane to be one of ['axial', 'coronal', 'sagittal']")



if __name__ == "__main__":

    n_classes = 2
    global_batch_size = 8
    volume_shape = (64, 64, 64)
    ds = get_dataset(
        ROOTDIR + "tfrecords/tfrecords_fold_1/data-train_*",
        n_classes=n_classes,
        batch_size=global_batch_size,
        volume_shape=volume_shape,
        plane="combined",
        shuffle_buffer_size=3,
    )

    print(ds)
    for _ in range(100):
        x,y=next(ds.as_numpy_iterator())
        print (np.min(x), np.max(x), np.unique(y))
        if np.max(x) == np.nan:
            print(x)
    """
    print (y)
    import matplotlib 
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = 5
    count = 1
    idx = np.random.randint(0, 32, rows)
    for i in range(rows):
        for j in range(3):
            plt.subplot(rows, 3, count)
            plt.imshow(x[idx[i], j, :, :, 0])
            plt.title(str(y[idx[i]]))
            count += 1
    plt.savefig("processed_image.png")
    """
# dataset_train_coronal = get_dataset("tfrecords/tfrecords_fold_1/data-train_*",
#                             n_classes=n_classes,
#                             batch_size=global_batch_size,
#                             volume_shape=volume_shape,
#                             block_shape=block_shape,
#                             plane='coronal',
#                             shuffle_buffer_size=3)

# dataset_train_sagittal = get_dataset("tfrecords/tfrecords_fold_1/data-train_*",
#                             n_classes=n_classes,
#                             batch_size=global_batch_size,
#                             volume_shape=volume_shape,
#                             block_shape=block_shape,
#                             plane='sagittal',
#                             shuffle_buffer_size=3)
