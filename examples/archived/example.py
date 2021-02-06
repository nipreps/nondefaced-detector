import sys
sys.path.append('..')
from defacing.helpers.utils import load_vol
from defacing.inference import inferer
from glob import glob

_inferer = inferer()
paths = glob('/work/01329/poldrack/data/mriqc-net/data/test_images/test1_images/*/*.nii.gz')

for path in paths:
    vol, _, _ = load_vol(path)
    label, conf = _inferer.infer(vol)

"""
_inferer = inferer()
path = '../sample_vols/faced/example4.nii.gz'
vol, _, _ = load_vol(path)
label, conf = _inferer.infer(vol)
"""
