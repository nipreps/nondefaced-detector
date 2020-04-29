import matplotlib 
# matplotlib.use('Agg')
import os, sys
sys.path.append('..')
from defacing.preprocessing.normalization import clip, standardize, normalize
from scipy.ndimage import map_coordinates
import numpy as np
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt


orig_data_face   = '../sample_vols/faced'
orig_data_face_mask = '../sample_vols/defaced'

save_data_face   = '../sample_vols/faced/conformed'
save_data_deface = '../sample_vols/defaced/conformed'

os.makedirs(save_data_face, exist_ok=True)
os.makedirs(save_data_deface, exist_ok=True)


conform_size = (64, 64, 64)
conform_zoom = (4., 4., 4.)

def preprocess(pth, mask_path=None, debug=False):
    """
    """
    filename = pth.split('/')[-1]
    in_file = nib.load(path)
    dtype = in_file.header.get_data_dtype()

    # Reorient to closest canonical
    in_file = nib.as_closest_canonical(in_file)

    # Calculate the factors to normalize voxel size to out_zooms
    normed = np.array(conform_zoom) / np.array(in_file.header.get_zooms()[:3])

    volume = np.array(in_file.dataobj)*1.0
    volume = clip(volume, q=95)
    volume = normalize(volume)
    volume = standardize(volume)
    
    
    save_volume_path = os.path.join(save_data_face, filename)
    
    def _plot(data):
        f, axarr = plt.subplots(8, 8, figsize=(12, 12))
        for i in range(8):
            for j in range(8):
                axarr[i, j].imshow(np.rot90(data[:, :, j + 8*i], 1))

        plt.show()
        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(np.mean(data, axis=0)))
        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(np.mean(data, axis=1)))
        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(np.mean(data, axis=2)))
        plt.show()


    def _conformation(data, save_path, order=3):
        new_ijk = normed[:, np.newaxis] * np.array(np.meshgrid(
            np.arange(conform_size[0]),
            np.arange(conform_size[1]),
            np.arange(conform_size[2]),
            indexing="ij")
        ).reshape((3, -1))
        offset = 0.5 * (np.max(new_ijk, axis=1) - np.array(in_file.shape))
        # Align the centers of the two sampling extents
        new_ijk -= offset[:, np.newaxis]

        # Resample data in the new grid
        resampled = map_coordinates(
            data,
            new_ijk,
            output=dtype,
            order=order,
            mode="constant",
            cval=0,
            prefilter=True,
        ).reshape(conform_size)
        resampled[resampled < 0] = 0
        
        if debug: _plot(resampled)

        # Create a new x-form affine, aligned with cardinal axes, 1mm3 and centered.
        newaffine = np.eye(4)
        newaffine[:3, 3] = -0.5 * (np.array(conform_size) - 1)
        nii = nib.Nifti1Image(resampled, newaffine, None)
        nii.to_filename(save_path)
        return save_path


    _conformation(volume, save_volume_path)
    if mask_path:
        mask = np.array(nib.load(mask_path).dataobj)
        masked_volume = volume*mask
        save_masked_path = os.path.join(save_data_deface, filename)
        _conformation(masked_volume, save_masked_path)
        
        return save_volume_path, save_masked_path
    return save_volume_path



for path in glob(orig_data_face+'/*.nii.gz'):
    # try:
    print(preprocess(path, debug=True))
    # except: pass

