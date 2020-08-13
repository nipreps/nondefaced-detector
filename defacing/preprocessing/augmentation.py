import tensorflow as tf
import numpy as np
from nobrainer.transform import get_affine
from nobrainer.transform import warp_features_labels


def zoom(x, shape=(64, 64)):
    """Zoom augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        crops = tf.image.crop_and_resize(
            [img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=shape
        )
        return crops[np.random.randint(0, len(scales))]

    choice = np.random.uniform(0, 1.0)
    if choice < 0.5:
        return lambda: x
    else:
        return lambda: random_crop(x)


class VolumeAugmentations(object):
    def __init__(self, DISTRIBUTION, augmentations=None):
        self.TRUEDISTRIBUTION = DISTRIBUTION
        self.distribution = DISTRIBUTION
        self.augmentations = augmentations

        allowed = ["rotation", "translation", "noop"]
        for key in augmentations.keys():
            if key.lower() not in allowed:
                raise ValueError(
                    "{} following augmentation not allowed. \n\
                                    Allowed augmentations are: {}".format(
                        key, allowed
                    )
                )

    def __call__(self, x, y):

        p = np.random.uniform(0, 1)
        if "noop" in self.augmentations.keys():
            if p < self.augmentations["noop"]:
                return (x, y)

        for key in self.augmentations.keys():
            if p < self.augmentations[key]:
                if key.lower() == "rotation":
                    k = np.random.randint(1, 4)
                    x = tf.image.rot90(x, k)
                    self.distribution = np.rot90(self.TRUEDISTRIBUTION, k)
                elif key.lower() == "translation":
                    matrix = get_affine(
                        volume_shape=np.asarray(x.shape, translation=translation)
                    )
                    x, self.distribution = warp_features_labels(
                        features=x, labels=self.distribution, matrix=matrix
                    )
                    self.distribution = tf.make_ndarray(self.distribution)

        return (x, y)


class SliceAugmentations(object):
    def __init__(self, augmentations=None):

        self.augmentations = augmentations

        allowed = ["rotation", "fliplr", "flipud", "zoom", "noop"]
        for key in augmentations.keys():
            if key not in allowed:
                raise ValueError(
                    "{} following augmentation not allowed. \n\
                                    Allowed augmentations are: {}".format(
                        key, allowed
                    )
                )

    def __call__(self, x):

        p = np.random.uniform(0, 1)
        if "noop" in self.augmentations.keys():
            if p < self.augmentations["noop"]:
                return x

        for key in self.augmentations.keys():
            if p < self.augmentations[key]:
                if key.lower() == "rotation":
                    k = np.random.randint(1, 4)
                    x = tf.image.rot90(x, k)

                elif key.lower() == "fliplr":
                    x = tf.image.random_flip_left_right(x)

                elif key.lower() == "flipud":
                    x = tf.image.random_flip_up_down(x)

                elif key.lower() == "zoom":
                    x = zoom(x, shape=np.asarray(x.shape)[-3:-1])

        return x
