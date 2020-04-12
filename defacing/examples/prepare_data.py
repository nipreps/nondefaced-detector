import os, sys
sys.path.append("..")
import binascii
from helpers.utils import load_vol, save_vol
from preprocessing.normalization import standardize_volume, normalize_volume
from preprocessing.conform import conform_data
import numpy as np
from glob import glob
from pathlib import Path
import nibabel as nib
from shutil import *
import subprocess


orig_data_face = "/work/01329/poldrack/data/mriqc-net/data/face/T1w"
orig_data_deface = "/work/01329/poldrack/data/mriqc-net/data/defaced"

save_data_face = "/work/06850/sbansal6/maverick2/mriqc-shared/face"
save_data_deface = "/work/06850/sbansal6/maverick2/mriqc-shared/deface"

os.makedirs(save_data_face, exist_ok=True)
os.makedirs(save_data_deface, exist_ok=True)


conform_size = (64, 64, 64)

def is_gz_file(filepath):
    if os.path.splitext(filepath)[1] == '.gz':
        with open(filepath, 'rb') as test_f:
            return binascii.hexlify(test_f.read(2)) == b'1f8b'

def preprocess(pth, conform_size):
    """
    """
    print(pth)
    filename = pth.split("/")[-1]
    print('Confirmation step')
    volume = conform_data(pth, out_size=conform_size)
    
    print("Normalize/Standardize step")
    volume = normalize_volume(standardize_volume(volume))
    save_path = os.path.join(save_data_face, 'conformed', filename)

    newaffine = np.eye(4)
    newaffine[:3, 3] = -0.5 * (np.array(conform_size) - 1)
    nii = nib.Nifti1Image(volume, newaffine, None)
    
    print("Save new affine")
    nii.to_filename(save_path)
    return save_path


for path in glob(orig_data_face + "/*/*.nii.gz"):
    print(path)
    if not is_gz_file(path):
        tempname = path.split("/")[-1]
        rename_file = os.path.splitext(tempname)[0]
        dst = os.path.join(save_data_face, rename_file)
        print(dst)
        # For some reason I was getting PermissionDenied when I tried to 
        # use shutil (which is why this hack)
        subprocess.call(['cp', path, dst])
        print(preprocess(dst, conform_size))
        subprocess.call(['rm', '-rf', dst])
    else:
        print(preprocess(path, conform_size))


# for path in glob(orig_data_deface + "/*/*.nii.gz"):
#     try:
#         print(preprocess(path))
#     except:
#         pass