import sys

sys.path.append("..")
from defacing.helpers.utils import load_volume
from defacing.inference import inferer

_inferer = inferer(threshold=0.7)
path = "../sample_vols/defaced/example1.nii.gz"
vol = load_volume(path)
label, conf = _inferer.infer(vol)
