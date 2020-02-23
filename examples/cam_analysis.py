import sys
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


volume_path = '../sample_vols/faced/example3.nii.gz'
volume = load_volume(volume_path)


model_path = '../defacing/saved_weights/best_cv2.h5'
model = custom_model(input_shape = (64,64),nclasses=2, multiencoders=True)
# model.load_weights(model_path, by_name=True)

print (model.summary())
dataloader_params = {'image_size': 64, 'nchannels': 1, 'nmontecarlo':1, 'transform':None}
datagenerator = DataGeneratoronFly(**dataloader_params)
X1, _, _ = datagenerator.get_data(volume)
X2 = np.array(X1[0])

save_path = '../cam_results'

for layer in range(1, len(model.layers)):
    print (X2[0, 0].shape, model.layers[layer].name)
    if model.layers[layer].name not in ['dense', 'dropout', 'output_node']: continue
    
    if save_path:
        plt.figure(figsize=(30, 20))
        gs = gridspec.GridSpec(2, 3)
        gs.update(wspace=0.025, hspace=0.05)

    for class_ in range(2):
        grads_ = visualize_cam(model, -1, filter_indices=class_, penultimate_layer_idx = 0, 
                                seed_input = X2)
        grads_ = np.array(grads_)
        print (X2.shape, grads_.shape)

