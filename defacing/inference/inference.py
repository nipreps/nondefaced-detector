import os
import sys
import numpy as np
import tensorflow as tf
import timeit

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from ..models.modelN import CombinedClassifier
from ..dataloaders.inference_dataloader import DataGeneratoronFly

ROOTDIR = "/work/06850/sbansal6/maverick2/mriqc-shared/"


class inferer(object):
    """
       nMontecarlo: for multiple exp for same model
       quick: checks for all 3 fold models
       mode: method to merge predictions
             allowed ['avg', 'max_vote']
    """

    def __init__(self, nMontecarlo=8, mode="avg"):
        r"""
        """
        inference_transform_params = {
            "conform_size": (64, 64, 64),
            "conform_zoom": (4., 4., 4.), 
            "nchannels": 1, 
            "nruns": 8,
            "nsamples": 20,
            "save": False, 
            "transform": None
        }

        self.mode = mode
        assert self.mode.lower() in [
            "avg",
            "max_vote",
        ], "unknown mode, allowed mode are ['avg', 'max_vote']"

        self.inference_generator = DataGeneratoronFly(**inference_transform_params)
        self.model = CombinedClassifier(
            input_shape=(64, 64), dropout=0.4, wts_root=None, trainable=True
        )
        self.model.load_weights(

            os.path.abspath(os.path.join(ROOTDIR, "model_save_dir_final/weights/combined/best-wts.h5"))
        )

    def infer(self, vol):
        """
        vol : can be numpy ndarray or path to volume
        """
        slices = self.inference_generator.get_data(vol)
        
        slices = np.transpose(np.array(slices),axes=[1, 0, 2, 3, 4])
        ds = {}
        ds['axial'] = slices[0]
        ds['coronal'] = slices[1]
        ds['sagittal'] = slices[2]
    
        predictions = self.model.predict(ds)

        if self.mode.lower() == "max_vote":
            predictions = np.round(predictions)
            unique_elements = np.unique(predictions)
            count_array = np.array(
                [
                    sum(predictions == unique_element)
                    for unique_element in unique_elements
                ]
            )
            pred = (
                np.argmax(count_array) if len(count_array) > 1 else unique_elements[0]
            )
            conf = (
                1
                if len(count_array) == 1
                else count_array[pred] * 1.0 / np.sum(count_array)
            )
        elif self.mode.lower() == "avg":
            conf = np.mean(predictions)
            pred = np.round(conf)

        pred_str = "faced" if pred == 1 else "defaced"
        conf = conf if pred == 1 else 1.0 - conf
        
        print("[INFO] Given volume is " + pred_str + " with confidence of: {}".format(conf))
        
        # del self.model
        # K.clear_session()
        return pred, conf
