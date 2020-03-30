import sys, os
import numpy as np
sys.path.append('..')
from defacing.helpers.utils import load_volume
from defacing.inference import inferer

root_path = '/home/pi/Downloads/IXI-PD'
_inferer = inferer(threshold=0.5)

ixi_volumes = os.listdir(root_path)
gt = np.array([1]*len(ixi_volumes))

labels, confs = [], []
for vol in ixi_volumes:
	path = os.path.join('/home/pi/Downloads/IXI-PD', vol)
	vol = load_volume(path)
	label, conf = _inferer.infer(vol)
	labels.append(label)
	confs.append(conf)

labels = np.array(labels)
tp = np.sum((gt == 1)*(labels ==1))
fp = np.sum((gt == 0)*(labels ==1))
tn = np.sum((gt == 0)*(labels ==0))
fn = np.sum((gt == 1)*(labels ==0))

print ("TP: {}, FP: {}, TN: {}, FN: {}".format(tp, fp, tn, fn))
