import sys, os
sys.path.append('..')
from defacing.helpers.utils import load_vol
from defacing.dataloaders.inference_dataloader import DataGeneratoronFly
from defacing.models.modelN import Submodel

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import activations
import vis
from vis.visualization import visualize_cam
from vis.utils import utils
import keras.backend as K
import tensorflow as tf

volume_path = '../sample_vols/faced/example1.nii.gz'
volume, _, _ = load_vol(volume_path)


root_path = '/home/pi/weights/'
weights = 'axial'
model = Submodel(input_shape=(64, 64),
                name=weights,
                weights=weights,
                include_top=True,
                root_path=root_path,
                trainable=False)



print (model.summary())
dataloader_params = {
            "conform_size": (64, 64, 64),
            "conform_zoom": (4., 4., 4.), 
            "nchannels": 1, 
            "nruns": 8,
            "nsamples": 20,
            "save": False, 
            "transform": None
        }
datagenerator = DataGeneratoronFly(**dataloader_params)
slices = datagenerator.get_data(volume)
slices = np.transpose(np.array(slices),axes=[1, 0, 2, 3, 4])
ds = {}
ds['axial'] = slices[0]
ds['coronal'] = slices[1]
ds['sagittal'] = slices[2]

score = np.squeeze(model.predict(ds[weights]))
print(score)
save_path = '../cam_results'

os.makedirs(save_path, exist_ok=True)
plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.025, hspace=0.05)

grads_ =visualize_cam(model, 0, 1, ds[weights], 
                      penultimate_layer_idx = -1)
grads_ = np.array(grads_)
ax = plt.subplot(gs[0,0])
im = ax.imshow(np.rot90(np.squeeze(ds[weights])), cmap='gray')
im = ax.imshow(np.rot90(grads_), alpha=1, cmap='jet')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal')
ax.tick_params(bottom='off', top='off', labelbottom='off' )
ax.set_title("confidence: {0:.2f}".format(100*float(score)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cb = plt.colorbar(im, ax=ax, cax=cax )

plt.savefig(os.path.join(save_path, 'cam_example3.png'), tight_box=True)
print (X2.shape, grads_.shape)

