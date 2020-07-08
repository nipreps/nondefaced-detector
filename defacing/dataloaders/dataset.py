import nobrainer
from nobrainer.io import _is_gzipped
from nobrainer.volume import to_blocks
import sys
sys.path.append('..')
from helpers.utils import load_vol
import tensorflow_probability as tfp
import tensorflow as tf
import glob
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
ROOTDIR = '/work/06850/sbansal6/maverick2/mriqc-shared/'
DISTRIBUTION = load_vol('../helpers/distribution.nii.gz')[0]
DISTRIBUTION /= DISTRIBUTION.sum()
COM = np.unravel_index(int(np.sum(DISTRIBUTION.ravel()*np.arange(len(DISTRIBUTION.ravel())))/np.sum(DISTRIBUTION.ravel())), DISTRIBUTION.shape)


# sampling from augmented distribution is same as augmenting the sampled points
# augmenting distribution at every iteration is expensive, so this way
sampler = lambda n, threshold = 0.1: np.array([ np.unravel_index(
          np.random.choice(np.arange(np.prod(DISTRIBUTION.shape)),
                                     p = DISTRIBUTION.ravel()),
          DISTRIBUTION.shape) + (+1 if np.random.randn() > 0.5 else -1)*np.random.randint(0, 
                                        int(DISTRIBUTION.shape[0]*threshold) + 1, 3) for _ in range(n)]) 


# function to apply augmentations to tf dataset
def apply_augmentations(x):

    """ Apply <TYPE_OF> augmentation to the dataset

    """
    p = np.random.uniform(0, 1, 1)
    if p < 0.33:
        # rotate 90
        x =  tf.image.rot90(
                   x, 1, name=None
                )
    elif p < 0.66:
        x =  tf.image.rot90(
                   x, 2, name=None
                )
    else:
        x =  tf.image.rot90(
                   x, 3, name=None
                )

    x = tf.keras.preprocessing.image.random_zoom(
                x, 0.2, 
                row_axis=1, 
                col_axis=2, 
                channel_axis=0, 
                fill_mode='nearest',
                cval=0.0, 
                interpolation_order=2
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


def get_dataset(
    file_pattern,
    n_classes,
    batch_size,
    volume_shape,
    plane,
    n = 24,
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

    # if augment:
    #     ds = ds.map(
    #         lambda x, y: tf.cond(
    #             tf.random.uniform((1,)) > 0.5,
    #             true_fn=lambda: apply_augmentations(x, y),
    #             false_fn=lambda: (x, y),
    #         ),
    #         num_parallel_calls=num_parallel_calls,
    #     )

    def _ss(x, y):
        x, y = structural_slice(x, y, plane, n, augment)
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


    ds = ds.prefetch(buffer_size=batch_size)
    def reshape(x,y):
        if plane == "combined":
            for _ in 3:
                pass
        return (x, y)
    if batch_size is not None:
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if shuffle_buffer_size:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat the dataset n_epochs times
    ds = ds.repeat(n_epochs)

    return ds


def structural_slice(x, y, plane, n=4, augment= False):

    """ Transpose dataset based on the plane

    Parameters
    ----------
    x:

    y:

    plane:
    
    n:

    augment:
    """

    threshold = 0.1 if augment else 0.0 
    options = ["axial", "coronal", "sagittal", "combined"]
    shape = np.array(x.shape)

    if isinstance(plane, str) and plane in options:
        if plane == "axial":
            idx = np.random.randint(shape[0]**0.5)
            midx = sampler(n, threshold)[:, 0]
            x = x
            k = 3

        if plane == "coronal":
            idx = np.random.randint(shape[1]**0.5)
            midx = sampler(n, threshold)[:, 1]
            x = tf.transpose(x, perm=[1, 2, 0])
            k = 2

        if plane == "sagittal":
            idx = np.random.randint(shape[2]**0.5)
            midx = sampler(n, threshold)[:, 2]
            x = tf.transpose(x, perm=[2, 0, 1])
            k = 1

        if plane == "combined":
            temp = {}
            for op in options[:-1]:
                temp[op] = structural_slice(x, y, op, n, augment)[0]
            x = temp

        if not plane == "combined":
            x = tf.squeeze(tf.gather_nd(x, midx.reshape(n, 1, 1)), axis=1)
            x = tf.math.reduce_mean(x, axis=0)
            x = tf.expand_dims(x, axis=-1)

            if augment:
                if np.random.uniform() > 0.3: 
                    # applies augmentation for 70% of the times
                    x = apply_augmentations(x)
                
            x = tf.convert_to_tensor(x)
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
