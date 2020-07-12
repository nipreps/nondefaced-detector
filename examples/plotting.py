import os
import numpy as np
from nilearn.image import smooth_img
from nilearn import plotting
from glob import glob
import sys
sys.path.append('..')
from defacing.helpers.utils import load_vol

ROOT_PATH = '/work/06850/sbansal6/maverick2/mriqc-shared/preprocessing'
# ROOT_PATH = '../'
SAVE_ROOT = '../save_plots/'
DISTRIBUTION = load_vol('../defacing/helpers/distribution.nii.gz')[0]
DISTRIBUTION /= DISTRIBUTION.sum()

sampler = lambda n: np.array([ np.unravel_index(
          np.random.choice(np.arange(np.prod(DISTRIBUTION.shape)),
                                      p = DISTRIBUTION.ravel()),
                                      DISTRIBUTION.shape) for _ in range(n)]) 


def plot(img_path, nslices=10, _class=None, dataset=None):
    save_path = os.path.join(SAVE_ROOT, _class, dataset, img_path.split('/')[-1].split('.')[0])
    os.makedirs(save_path, exist_ok=True)
    coordinates = sampler(10)
    img = smooth_img(img_path, fwhm=3) 
    for i, coordinate in enumerate(coordinates):
        display = plotting.plot_anat(img, display_mode='ortho', cut_coords = coordinate) 
        display.savefig(os.path.join(save_path, str(i)+'.png'))
    display.close()
    pass

def plot_all(dataset, 
             _class, 
             nvolumes = 25, 
             nslices = 10, 
             verbose = True):
    
    paths = glob(os.path.join(ROOT_PATH, _class, dataset, '*.nii.gz'))
    paths = np.random.choice(paths, size=nvolumes) if len(paths) > nvolumes else paths
    
    for path in paths:
        print("="*10 + path + "="*10)
        plot(path, nslices = nslices, _class = _class, dataset = dataset)
    pass

if __name__ == "__main__":
    classes = os.listdir(ROOT_PATH)
    for class_ in classes:
        datasets = os.listdir(os.path.join(ROOT_PATH, class_))
        for ds in datasets:
            print ("[INFO]: Dataset {}, class_ {}".format(ds, class_))
            plot_all(ds, class_, nvolumes = 20, nslices = 10)
