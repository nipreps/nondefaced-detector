import nobrainer
from nobrainer.io import _is_gzipped
from nobrainer.volume import to_blocks
import sys, os
sys.path.append('..')
from preprocessing.augmentation import VolumeAugmentations, SliceAugmentations
from helpers.utils import load_vol
import tensorflow as tf
import glob
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
NAME = 'distribution.nii.gz'

for path in sys.path:
    for root, dirs, files in os.walk(path):
        if NAME in files:
            DISTRIBUTION = load_vol(os.path.join(root, NAME))[0]
            break
            
DISTRIBUTION /= DISTRIBUTION.sum()
COM = np.unravel_index(int(np.sum(DISTRIBUTION.ravel()*np.arange(len(DISTRIBUTION.ravel())))/np.sum(DISTRIBUTION.ravel())), DISTRIBUTION.shape)


# sampling from augmented distribution is same as augmenting the sampled points
# augmenting distribution at every iteration is expensive, so this way
sampler = lambda n_slices, distribution = DISTRIBUTION, threshold = 0.1: np.array([ np.unravel_index(
          np.random.choice(np.arange(np.prod(distribution.shape)),
                                     p = distribution.ravel()),
          distribution.shape) + (+1 if np.random.randn() > 0.5 else -1)*np.random.randint(0, 
                                        int(distribution.shape[0]*threshold) + 1, 3) for _ in range(n_slices)]) 


three_d_augmentations = {'rotation': 0.5,
                         'translation': 0.5,
                         'noop': 0.3
                        }

augmentvolume = VolumeAugmentations(DISTRIBUTION, three_d_augmentations)

two_d_augmentations = {'rotation': 0.5,
                       'fliplr': 0.5,
                       'flipud': 0.5,
                       'zoom': 0.5,
                       'noop': 0.3
                      }

# augmentslice = VolumeAugmentations(DISTRIBUTION, two_d_augmentations)


def get_dataset(
    file_pattern,
    n_classes,
    batch_size,
    volume_shape,
    plane,
    n_slices = 24,
    block_shape=None,
    n_epochs=None,
    mapping=None,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=AUTOTUNE,
    mode='train',
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
        if augment:
            if three_d_augmentations['noop'] < 1:
                x, y = augmentvolume(x,y)
        x, y = structural_slice(x, y, 
                                plane, 
                                n_slices, 
                                augment, 
                                augmentvolume.distribution)
        return (x, y)
    
    
    ds = ds.map(_ss, num_parallel_calls)
    
    ds = ds.prefetch(buffer_size=batch_size)

    if batch_size is not None:
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        
    if mode == 'train':
        if shuffle_buffer_size:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat the dataset n_epochs times
        ds = ds.repeat(n_epochs)

    return ds


def structural_slice(x, y, 
                plane, 
                n_slices = 4, 
                augment = False, 
                distribution = DISTRIBUTION):

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
        idxs = sampler(n_slices, 
                        distribution, 
                        threshold)

        if plane == "axial":
            idx = np.random.randint(shape[0]**0.5)
            midx = idxs[:, 0]
            x = x

        if plane == "coronal":
            idx = np.random.randint(shape[1]**0.5)
            midx = idxs[:, 1]
            x = tf.transpose(x, perm=[1, 2, 0])


        if plane == "sagittal":
            idx = np.random.randint(shape[2]**0.5)
            midx = idxs[:, 2]
            x = tf.transpose(x, perm=[2, 0, 1])


        if plane == "combined":
            temp = {}
            for op in options[:-1]:
                temp[op] = structural_slice(x, y, 
                                            op, 
                                            n_slices, 
                                            augment, 
                                            distribution)[0]
            x = temp

        if not plane == "combined":
            x = tf.squeeze(tf.gather_nd(x, midx.reshape(n_slices, 1, 1)), axis=1)
            x = tf.math.reduce_mean(x, axis=0)
            x = tf.expand_dims(x, axis=-1)
            
            if augment:
                x = two_d_augmentations(x)
                
            x = tf.convert_to_tensor(x)
        return x, y
    else:
        raise ValueError("expected plane to be one of ['axial', 'coronal', 'sagittal']")


if __name__ == "__main__":
    ROOTDIR = '/home/shank/HDDLinux/Stanford/data/mriqc-shared/experiments/experiment_B/128/tfrecords_full'
    n_classes = 2
    global_batch_size = 8
    volume_shape = (64, 64, 64)
    ds = get_dataset(
        os.path.join(ROOTDIR, "data-train_*"),
        n_classes=n_classes,
        batch_size=global_batch_size,
        volume_shape=volume_shape,
        plane="sagittal",
        augment = False,
        shuffle_buffer_size=3,
    )

    print(ds)