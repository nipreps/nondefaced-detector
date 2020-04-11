import os, sys
sys.path.append('..')
from defacing.helpers.utils import load_vol, save_vol
import numpy as np
from glob import glob

data_face   = '/work/01329/poldrack/data/mriqc-net/face/derivatives/conformed'
data_deface = '/work/01329/poldrack/data/mriqc-net/defaced/derivatives/conformed'
tfrecord_path = '/work/01329/poldrack/data/mriqc-net/tfrecords/deidentification'
os.makedirs(tfrecord_path, exist_ok=True)

def extract_slices(volume, nsamples = 1024):
    """

    """
    dimensions = volume.shape
    x = np.random.randint(dimensions[0]//3, 2*dimensions[0]//3, nsamples)
    z = np.random.randint(dimensions[1]//4, dimensions[1]//2, nsamples)
    y = np.random.randint(dimensions[2]//3, dimensions[2]//4, nsamples)
    
    slices = (volume[x, :, :][:, None, ..., None], 
               volume[:, y, :][:, None, ..., None], 
               volume[:, :, z][:, None, ..., None]) # 3x nsamples x 1 x 64 x 64x 1
    slices = np.concatenate(slices, axis=1) # nsamples x 3 x 64x 64x 1
    return slices

paths = glob(data_face+'/*.nii.gz')
paths.extend(glob(data_deface+'/*nii.gz')

xdata, ydata = [], []

for pth in paths:
    nsamples = 512
    volume = load_vol(pth)
    slices = extract_slices(volume, nsamples)
    label  = 1 if pth.__contains__('defaced') else 0
    xdata.extend(slices)
    ydata.extend([label]*nsamples)

# generate tfrecords
