import os
import sys
import time
import imgaug
import shutil
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from ..helpers.utils import save_vol, load_vol
from ..preprocessing.conform import conform_data
from ..preprocessing.normalization import clip, standardize, normalize


class DataGeneratoronFly(object):
    """
    """

    def __init__(self, conform_size=(64, 64, 64),
                        conform_zoom=(4., 4., 4.), 
                        nchannels=1, 
                        nruns=8,
                        nsamples=20,
                        save=False, 
                        transform=None):

        self.conform_size=conform_size
        self.conform_zoom=conform_zoom
        self.nchannels=nchannels
        self.transform=transform
        self.nsamples=nsamples
        self.nruns=nruns
        self.save=save
        DISTRIBUTION = load_vol('../defacing/helpers/distribution.nii.gz')[0]
        assert DISTRIBUTION.shape == conform_size, "Invalid conform_size needs to regenerate face distribution"

        DISTRIBUTION /= DISTRIBUTION.sum()
        self.sampler = lambda n: np.array([ np.unravel_index(
                  np.random.choice(np.arange(np.prod(DISTRIBUTION.shape)),
                                             p = DISTRIBUTION.ravel()),
                  DISTRIBUTION.shape) for _ in range(n)]) 



    def _augmentation(self, volume):
        r"""
                Augmenters that are safe to apply to masks
                Some, such as Affine, have settings that make them unsafe, so always
                test your augmentation on masks
        """
        volume_shape = volume.shape
        det = self.transform.to_deterministic()
        volume = det.augment_image(volume)

        assert volume.shape == volume_shape, "Augmentation shouldn't change volume size"
        return volume


    def _sample_slices(self, volume, plane=None):

        options = ["axial", "coronal", "sagittal", "combined"]
        assert plane in options, "expected plane to be one of ['axial', 'coronal', 'sagittal']"
        samples = self.sampler(self.nsamples)

        if plane == "axial":
            midx = samples[:, 0]
            volume = volume
            k = 3

        if plane == "coronal":
            midx = samples[:, 1]
            volume = np.transpose(volume, axes=[1, 2, 0])
            k = 2

        if plane == "sagittal":
            midx = samples[:, 2]
            volume = np.transpose(volume, axes=[2, 0, 1])
            k = 1

        if plane == "combined":
            temp = []
            for op in options[:-1]:
                temp.append(self._sample_slices(volume, op))
            volume = temp

        if not plane == "combined":
            volume = np.squeeze(volume[midx,:,:])
            volume = np.mean(volume, axis=0)
            volume = np.rot90(volume, k)
            volume = volume[..., None]
        return volume


    def get_data(self, volume):
        # Generate indexes of the batch
        volume = clip(volume, q=90)
        volume = normalize(volume)
        volume = standardize(volume)
        newaffine = np.eye(4)
        newaffine[:3, 3] = -0.5 * (np.array(self.conform_size) - 1)
        os.makedirs('./tmp', exist_ok=True)
        save_vol('./tmp/Pre-processing.nii.gz', volume, newaffine)
        conform_data('./tmp/Pre-processing.nii.gz',
                        './tmp/Conformed.nii.gz',
                        self.conform_size,
                        self.conform_zoom)

        volume = load_vol('./tmp/Conformed.nii.gz')[0]

        if self.transform:
            volume = self._augmentation(volume)

        slices = []
        for _ in range(self.nruns):
            slices.append(self._sample_slices(volume, 
                                    plane="combined"))

        if not self.save: 
            shutil.rmtree('./tmp')
        return slices
