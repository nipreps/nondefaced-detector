import sys
sys.path.append('..')
from defacing.helpers.utils import load_volume
from defacing.inference import inferer

inferer = inferer()
path = '../sample_vols/faced/example5.nii.gz'
vol = load_volume(path)
label = inferer.infer(vol)
