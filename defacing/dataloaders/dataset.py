import nobrainer
from nobrainer.io import _is_gzipped
from nobrainer.volume import to_blocks
import tensorflow_probability as tfp
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




def clip(x, q =90):
    """
    """
    min_val = 0
    max_val = tfp.stats.percentile(
                x, q, axis=None,
                preserve_gradients=False,
                name=None
                )
    x = tf.clip_by_value(
        x,
        min_val,
        max_val,
        name=None
        )
    return x

def _magic_slicing_(shape):
    """
    """
    idx = []
    for ii in np.arange(shape[0]):
        if (ii % shape[0]**0.5) == 0:
            idx.append(ii)
    idx = np.array(idx)
    return idx


def standardize(x):
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
    median = tfp.stats.percentile(
                x, 50, axis=None,
                preserve_gradients=False,
                name=None
                )
    mean, var = tf.nn.moments(x, axes=None)
    std = tf.sqrt(var)
    return (x - median) / std


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
    def reshape(x,y):
        if plane == "combined":
            for _ in 3:
                pass
        return (x, y)
    if batch_size is not None:
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        # ds = ds.map(lambda x,y: (tf.reshape(x, ((3, batch_size*n,) if plane == "combined" else (batch_size*n,)) + volume_shape[:2] +(1,)),
        #                         tf.reshape(y, (batch_size*n,))))

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
    mean_idx = _magic_slicing_(shape)

    x = clip(x)
    x = normalize(x)
    x = standardize(x)

    if isinstance(plane, str) and plane in options:
        if plane == "axial":
            idx = np.random.randint(int(shape[0]**0.5))
            x = x
            k = 3

        if plane == "coronal":
            idx = np.random.randint(int(shape[1]**0.5))
            x = tf.transpose(x, perm=[1, 2, 0])
            k = 2

        if plane == "sagittal":
            idx = np.random.randint(int(shape[2]**0.5))
            x = tf.transpose(x, perm=[2, 0, 1])
            k = 1

        if plane == "combined":
            temp = {}
            for op in options[:-1]:
                temp[op] = structural_slice(x, y, op, n)[0]
            x = temp

        if not plane == "combined":
            x = tf.squeeze(tf.gather_nd(x, (mean_idx + idx).reshape(int(shape[0]**0.5), 1, 1)), axis=1)
            x = tf.math.reduce_mean(x, axis=0)
            x = tf.convert_to_tensor(tf.expand_dims(x, axis=-1))
            x =  tf.image.rot90(
                   x, k, name=None
                )

        # y = tf.repeat(y, n)

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

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(ds)
    for ii in range(100):
        x,y=next(ds.as_numpy_iterator())
        # print (np.min(x), np.max(x), np.unique(y))
        count = 1
        for i in range(global_batch_size):
            for key in x.keys():
                plt.subplot(global_batch_size, 3, count)
                plt.imshow(x[key][i, :, :, 0])
                plt.title(str(y[i]))
                plt.xticks([]," ")
                plt.yticks([], " ")
                count += 1
        plt.savefig("processed_image_combined_{}.png".format(ii))


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
