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


orig_data_mask   = '/work/01329/poldrack/data/mriqc-net/data/masks'
orig_data_faced   = '/work/06850/sbansal6/maverick2/mriqc-shared/conformed/faced'
save_conformed_mask = '/work/06850/sbansal6/maverick2/mriqc-shared/masks/conformed'
saveprob_distribution = '../defacing/helpers/distribution.nii.gz'

os.makedirs(save_conformed_mask, exist_ok=True)


conform_size = (64, 64, 64)
probability_distribution = np.zeros(conform_size)
normalization = 0

conform_zoom = (4., 4., 4.)


for path in glob(orig_data_mask+'/*/*.nii.gz'):
    if 'ds000140_anat' in path: continue
    filename = path.split('/')[-1]
    dataset = path.split('/')[-2]
    try:
        conform_path = os.path.join(save_conformed_mask, dataset)
        mask_path = glob(os.path.join(orig_data_mask, dataset, filename.split('.')[0] + "*_mask*"))[0]
        volume_path = os.path.join(orig_data_faced, dataset, filename)
    except: continue
    os.makedirs(conform_path, exist_ok=True)   
    print(mask_path)
    conform_data(mask_path, 
                 out_file=os.path.join(conform_path,filename), 
                 out_size=conform_size, 
                 out_zooms=conform_zoom,
                 order=0)
    print("conform")
    volume = load_vol(volume_path)[0]
    volume = normalize(volume)

    # thresholding
    volume[volume>0.1] = 1; volume[volume<=0.1] = 0
    mask = load_vol(os.path.join(conform_path, filename))[0]
    mask = volume[(volume==1)*(mask==0)]
    normalization += 1
    probability_distribution += mask*1.
    print (path, np.min(probability_distribution), np.max(probability_distribution))
    # except: pass

# normalize probability
probability_distribution /= np.sum(probability_distribution)
affine = np.eye(4)
affine[:3, 3] = -0.5 * (np.array(conform_size) - 1)
save_vol(saveprob_distribution, probability_distribution/(normalization*1.), affine)
