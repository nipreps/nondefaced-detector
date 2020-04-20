import os, sys
sys.path.append('..')
from defacing.helpers.utils import load_vol, save_vol
from defacing.preprocessing.normalization import standardize_volume, normalize_volume
from defacing.preprocessing.conform import conform_data
import numpy as np
import nibabel as nib
from glob import glob

orig_data_face   = '../sample_vols/faced'
orig_data_deface = '../sample_vols/defaced'

save_data_face   = '../sample_vols/faced/conformed'
save_data_deface = '../sample_vols/defaced/conformed'
os.makedirs(save_data_face, exist_ok=True)
os.makedirs(save_data_deface, exist_ok=True)


conform_size = (64, 64, 64)

def preprocess(pth):
    """
    """
    filename = pth.split('/')[-1]
    volume = conform_data(pth, out_size = conform_size)

    volume = normalize_volume(standardize_volume(volume))    
    save_path = os.path.join(save_data_face, filename)

    newaffine = np.eye(4)
    newaffine[:3, 3] = -0.5 * (np.array(conform_size) - 1)
    nii = nib.Nifti1Image(volume, newaffine, None)
    nii.to_filename(save_path)
    return save_path

for path in glob(orig_data_face+'/*.nii.gz'):
    # try:
    print(preprocess(path))
    # except: pass

for path in glob(orig_data_deface+'/*.nii.gz'):
    # try:
    print(preprocess(path))
    # except: pass
