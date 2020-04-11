import os
import sys
import time
import imgaug
from imgaug import augmenters as iaa
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from .registration import Coregistration
from ..helpers.utils import *
from ..preprocessing.conform import conform_data


class DataGeneratoronFly(object):
    """
    """

    def __init__(self, image_size=64, nchannels=1, nmontecarlo=8, transform=None):

        self.image_size = image_size
        self.nchannels = nchannels
        self.transform = transform
        self.nmontecarlo = nmontecarlo
        self.coreg = Coregistration()
        fix_path = os.path.abspath("../defacing/helpers/fixed_image.nii.gz")
        self.fixed_image = load_volume(fix_path)

    def get_data(self, data, all=False):
        # Generate indexes of the batch
        # data = self.coreg.register_patient(
        #    data, self.fixed_image).astype('float64')
        _, data = conform_data(data)
        if all:
            return self.__data_generation(data, all=all)

        X1, X2, X3 = self.__data_generation(data)
        return [X1, X2, X3]

    def _standardize_volume(self, volume, mask=None):
        """
                volume: volume which needs to be normalized
                mask: brain mask, only required if you prefer not to
                        consider the effect of air in normalization
        """
        if mask != None:
            volume = volume * mask

        mean = np.mean(volume[volume != 0])
        std = np.std(volume[volume != 0])

        return (volume - mean) / std

    def _normalize_volume(self, volume, mask=None, _type="MinMax"):
        """
                volume: volume which needs to be normalized
                mask: brain mask, only required if you prefer not to
                        consider the effect of air in normalization
                _type: {'Max', 'MinMax', 'Sum'}
        """
        if mask != None:
            volume = mask * volume
        min_vol = np.min(volume)
        max_vol = np.max(volume)
        sum_vol = np.sum(volume)

        if _type == "MinMax":
            return (volume - min_vol) / (max_vol - min_vol)
        elif _type == "Max":
            return volume / max_vol
        elif _type == "Sum":
            return volume / sum_vol
        else:
            raise ValueError(
                "Invalid _type, allowed values are: {}".format("Max, MinMax, Sum")
            )

    def _augmentation(self, volume):
        """
                Augmenters that are safe to apply to masks
                Some, such as Affine, have settings that make them unsafe, so always
                test your augmentation on masks
        """
        volume_shape = volume.shape
        det = self.transform.to_deterministic()
        volume = det.augment_image(volume)

        assert volume.shape == volume_shape, "Augmentation shouldn't change volume size"
        return volume

    def _resizeVolume(self, volume):
        """
                resizes the original volume such that every patch is
                75% of original volume

                volume: numpy 3d tensor
        """
        ratio = 1.0

        orig_size = (
            int(self.image_size / ratio),
            int(self.image_size / ratio),
            int(self.image_size / ratio),
        )
        resized_volume = resize_sitk(volume, orig_size)
        return resized_volume

    def _get_random_slices(self, volume):
        """
        """
        dimensions = volume.shape
        x = np.random.randint(
            dimensions[0] // 5, 4 * dimensions[0] // 5, self.nmontecarlo
        )
        z = np.random.randint(
            dimensions[1] // 5, 4 * dimensions[1] // 5, self.nmontecarlo
        )
        y = np.random.randint(
            dimensions[2] // 5, 4 * dimensions[2] // 5, self.nmontecarlo
        )

        return [
            volume[x, :, :][..., None],
            volume[:, y, :].transpose(1, 0, 2)[..., None],
            volume[:, :, z].transpose(2, 0, 1)[..., None],
        ]

    def _get_all_slices(self, volume):
        """
        """
        dimensions = volume.shape
        x = list(range(dimensions[0] // 5, 4 * dimensions[0] // 5))
        z = list(range(dimensions[1] // 5, 4 * dimensions[1] // 5))
        y = list(range(dimensions[2] // 5, 4 * dimensions[2] // 5))

        return [
            volume[x, :, :][..., None],
            volume[:, y, :].transpose(1, 0, 2)[..., None],
            volume[:, :, z].transpose(2, 0, 1)[..., None],
        ]

    def __data_generation(self, volume, all=False):
        """
        """

        volume = self._resizeVolume(volume)
        volume = self._standardize_volume(volume)
        volume = self._normalize_volume(volume)

        if self.transform:
            volume = self._augmentation(volume)

        if all:
            return self._get_all_slices(volume)

        X1 = self._get_random_slices(volume)
        X2 = self._get_random_slices(volume)
        X3 = self._get_random_slices(volume)

        return X1, X2, X3
