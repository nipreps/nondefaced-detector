import os, sys
import numpy as np
import tensorflow as tf
import timeit

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from ..models.model import custom_model
from ..dataloaders.inference_dataloader import DataGeneratoronFly

class inferer(object):
    """
       nMontecarlo: for multiple exp for same model
       quick: checks for all 3 fold models
       mode: method to merge predictions
             allowed ['avg', 'max_vote']
    """
    def __init__(self, 
                nMontecarlo=8,
                quick = False,
                mode = 'avg'):
 

        inference_transform_params = {'image_size': 64,
                              'nchannels': 1,
                              'nmontecarlo': nMontecarlo,
                              'transform': None
                             }
        self.quick = quick
        self.mode  = mode
        assert self.mode.lower() in ['avg', 'max_vote'], "unknown mode, allowed mode are ['avg', 'max_vote']"

        self.inference_generator = DataGeneratoronFly(**inference_transform_params)
        self.model_fold1 = custom_model(input_shape = (64, 64),
					nclasses = 2,
					multiencoders = True)
        self.model_fold1.load_weights(os.path.abspath('../defacing/saved_weights/best_cv1.h5'))

        if not self.quick: 
            self.model_fold2 = custom_model(input_shape = (64, 64),
					nclasses = 2,
					multiencoders = True)
            self.model_fold2.load_weights('../defacing/saved_weights/best_cv2.h5')


            self.model_fold3 = custom_model(input_shape = (64, 64),
					nclasses = 2,
					multiencoders = True)
            self.model_fold3.load_weights('../defacing/saved_weights/best_cv3.h5')


    def infer(self, vol):
        """
        vol : canbe numpy ndarray or path to volume
        """
        X1, X2, X3 = self.inference_generator.get_data(vol)
        predictions = self.model_fold1.predict(X1)
        
        if not self.quick:    	
            prediction_fold2 = self.model_fold2.predict(X2)    	
            prediction_fold3 = self.model_fold3.predict(X3)

            predictions = np.squeeze(np.concatenate([predictions, 
                                           prediction_fold2,
                                           prediction_fold3], axis=0)) 
        if self.mode.lower() == 'max_vote':   	
            predictions = np.round(predictions)
            unique_elements = np.unique(predictions)
            count_array = np.array([sum(predictions == unique_element) for unique_element in unique_elements])
            pred = np.argmax(count_array) if len(count_array) > 1 else unique_elements[0]
            conf = 1 if len(count_array) == 1 else count_array[pred]*1.0/np.sum(count_array)
        elif self.mode.lower() == 'avg':
            conf = np.mean(predictions)
            pred = np.round(conf)
        
        pred_str = 'faced' if pred == 1 else 'defaced'
        conf = conf if pred == 1 else 1.0 - conf
        print("Given volume is " + pred_str + " with confidence of: {}".format(conf)) 
        return pred, conf
