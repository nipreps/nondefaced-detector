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


orig_data_mask   = '/mask/01329/poldrack/data/mriqc-net/data/masks'
save_conformed_mask = '/work/06850/sbansal6/maverick2/mriqc-shared/masks/conformed'
saveprob_distribution = '../defacing/helpers/distribution.nii.gz'

os.makedirs(save_conformed_mask, exist_ok=True)


conform_size = (64, 64, 64)
probability_distribution = np.zeros(conform_size)
normalization = 0

conform_zoom = (4., 4., 4.)


for path in glob(orig_data_mask+'/*.nii.gz'):
    # try:
    filename = pth.split('/')[-1]
    dataset = pth.split('/')[-2]
    conform_path = os.path.join(save_conformed_mask, dataset, filename)
    conform_data(path, 
                 out_file=conformed_path, 
                 out_size=conform_size, 
                 out_zooms=conform_zoom)
    conformed_mask = load_vol(conform_path)[0]
    normalization += 1
    probability_distribution += conformed_mask*1.
    # except: pass

# normalize probability
probability_distribution /= np.sum(probability_distribution)
affine = np.eye(4)
affine[:3, 3] = -0.5 * (np.array(conform_size) - 1)
save_vol(saveprob_distribution, probability_distribution/(normalization*1.), affine)
