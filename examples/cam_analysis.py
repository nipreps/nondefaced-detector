import sys, os
sys.path.append('..')
from defacing.helpers.utils import load_volume
from defacing.dataloaders.inference_dataloader import DataGeneratoronFly
from defacing.models.model import custom_model

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
volume = load_volume(volume_path)


model_path = '../defacing/saved_weights/best_cv2.h5'
model = custom_model(input_shape = (64,64), nclasses=2, multiencoders=True)
session = K.get_session()
init = tf.global_variables_initializer()
session.run(init)
model.load_weights(model_path, by_name=True)

print (model.summary())
dataloader_params = {'image_size': 64, 'nchannels': 1, 'nmontecarlo':1, 'transform':None}
datagenerator = DataGeneratoronFly(**dataloader_params)
X1, _, _ = datagenerator.get_data(volume)
X2 = np.array(X1[2])
score = np.squeeze(model.predict(X2))
print(score)
save_path = '../cam_results'

    
if save_path:
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.025, hspace=0.05)

grads_ = visualize_cam(model, -1, filter_indices=0, penultimate_layer_idx = 0, 
                       seed_input = X2)
grads_ = np.array(grads_)
ax = plt.subplot(gs[0,0])
im = ax.imshow(np.rot90(np.squeeze(X2)), cmap='gray')
im = ax.imshow(np.rot90(grads_), alpha=0.5, cmap='jet')
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

