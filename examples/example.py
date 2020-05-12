import sys
sys.path.append('..')
from defacing.helpers.utils import load_vol
from defacing.inference import inferer

_inferer = inferer()
path = '../sample_vols/faced/example4.nii.gz'
vol, _, _ = load_vol(path)
label, conf = _inferer.infer(vol)

