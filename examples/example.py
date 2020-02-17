import sys
sys.path.append('..')
from defacing.helpers.utils import load_volume
from defacing.inference import inferer

_inferer = inferer(threshold=0.95)
path = '../sample_vols/defaced/example2.nii.gz'
vol = load_volume(path)
label = _inferer.infer(vol)
