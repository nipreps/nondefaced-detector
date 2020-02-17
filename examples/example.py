from defacing.helpers.utils import load_volume
from defacing.inference import inferer
import sys
sys.path.append('..')

_inferer = inferer(threshold=0.95)
path = '../sample_vols/faced/example1.nii.gz'
vol = load_volume(path)
label = _inferer.infer(vol)
