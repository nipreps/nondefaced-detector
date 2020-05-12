import os
import sys
import time
import imgaug
import numpy as np
from imgaug import augmenters as iaa
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

        DISTRIBUTION = load_vol('../helpers/distribution.nii.gz')[0]
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
        samples = sampler(self.nsamples)

        if plane == "axial":
            midx = samples[:, 0]
            volume = volume
            k = 3

        if plane == "coronal":
            midx = samplers[:, 1]
            volume = np.transpose(volume, axes=[1, 2, 0])
            k = 2

        if plane == "sagittal":
            midx = samplers[:, 2]
            volume = np.transpose(volume, axes=[2, 0, 1])
            k = 1

        if plane == "combined":
            temp = {}
            for op in options[:-1]:
                temp[op] = self._sample_slices(x, op)
            volumes = temp

        if not plane == "combined":
            x = np.squeeze(volume[midx,:,:])
            x = np.mean(volume, axis=0)
            x = np.rot90(x, k)
            x = x[..., None]
        return x


    def get_data(self, volume):
        # Generate indexes of the batch
        
        volume = clip(volume, q=90)
        volume = normalize(volume)
        volume = standardize(volume)
        newaffine = np.eye(4)
        newaffine[:3, 3] = -0.5 * (np.array(out_size) - 1)
        save_vol('Pre-processing.nii.gz', volume, newaffine)
        conform_data('Pre-processing.nii.gz',
                        'Conformed.nii.gz',
                        self.conform_size,
                        self.conform_zoom)

        volume = load_vol('Conformed.nii.gz')[0]

        if self.transform:
            volume = self._augmentation(volume)

        slices = []
        for _ in range(self.nruns):
            slices.append(self._sample_slices(volume, 
                                    plane="combined"))

        if not self.save: 
            os.remove('Pre-processing.nii.gz') 
            os.remove('Conformed.nii.gz') 
        return slices