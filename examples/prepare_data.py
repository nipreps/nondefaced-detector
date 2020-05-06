import matplotlib
# matplotlib.use('Agg')
import os, sys
sys.path.append('..')
from defacing.preprocessing.normalization import clip, standardize, normalize
from defacing.preprocessing.conform import conform_data
from defacing.helpers.utils import load_vol, save_vol
import numpy as np
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt


orig_data_face   = '../sample_vols/faced'
orig_data_face_mask = '../sample_vols/faced'

save_preprocessing_face   = '../sample_vols/faced/preprocessing'
save_conformed_face   = '../sample_vols/faced/conformed'

save_preprocessing_deface   = '../sample_vols/faced/preprocessing'
save_conformed_deface = '../sample_vols/defaced/conformed'

os.makedirs(save_preprocessing_face, exist_ok=True)
os.makedirs(save_preprocessing_deface, exist_ok=True)
os.makedirs(save_conformed_face, exist_ok=True)
os.makedirs(save_conformed_deface, exist_ok=True)

conform_size = (64, 64, 64)
conform_zoom = (4., 4., 4.)

def preprocess(pth, mask_path=None, debug=False):
    """
    """
    filename = pth.split('/')[-1]
    volume, affine, _ = load_vol(pth)

    volume = clip(volume, q=90)
    volume = normalize(volume)
    volume = standardize(volume)
    
    save_preprocessing_path = os.path.join(save_preprocessing_face, filename)
    save_conformed_path = os.path.join(save_conformed_face, filename)    
    save_vol(save_preprocessing_path, volume, affine)

    def _plot(data):
        f, axarr = plt.subplots(8, 8, figsize=(12, 12))
        for i in range(8):
            for j in range(8):
                axarr[i, j].imshow(np.rot90(data[:, :, j + 8*i], 1))

        plt.show()
        
        """
        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(np.mean(data, axis=0)))
        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(np.mean(data, axis=1)))
        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(np.mean(data, axis=2)))
        plt.show()
        """
    conform_data(save_preprocessing_path, 
                 out_file=save_conformed_path, 
                 out_size=conform_size, 
                 out_zooms=conform_zoom)

    if debug: _plot(load_vol(save_conformed_path)[0])
    
    if mask_path:
        mask = np.array(nib.load(mask_path).dataobj)
        masked_volume = volume*mask

        save_mpreprocessing_path = os.path.join(save_preprocessing_deface, filename)
        save_mconformed_path = os.path.join(save_conformed_deface, filename)    
        save_vol(save_mpreprocessing_path, masked_volume, affine)

        conform_data(save_mpreprocessing_path, 
                 out_file=save_mconformed_path, 
                 out_size=conform_size, 
                 out_zooms=conform_zoom)        
        return save_conformed_path, save_mconformed_path

    return save_conformed_path



for path in glob(orig_data_face+'/*.nii.gz'):
    # try:
    print(preprocess(path, debug=True))
    # except: pass

