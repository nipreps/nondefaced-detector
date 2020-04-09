import os
import sys
import numpy as np
import tensorflow as tf
import timeit

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from ..models.modelN import CombinedClassifier
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
                 threshold=0.7,
                 mode='avg'):
        inference_transform_params = {'image_size': 64,
                                      'nchannels': 1,
                                      'nmontecarlo': nMontecarlo,
                                      'transform': None
                                      }
        self.quick = quick
        self.mode = mode
        self.threshold = threshold

        assert self.mode.lower() in [
            'avg', 'max_vote'], "unknown mode, allowed mode are ['avg', 'max_vote']"

        self.inference_generator = DataGeneratoronFly(
            **inference_transform_params)
        self.model_fold1 = CombinedClassifier(input_shape=(32, 32), 
                                               dropout=0.4, 
                                               wts_root = None, 
                                               trainable = True)
        self.model_fold1.load_weights(os.path.abspath(
            '../defacing/saved_weights/best.h5'))


    def infer(self, vol):
        """
        vol : can be numpy ndarray or path to volume
        """
        X1, X2, X3 = self.inference_generator.get_data(vol)
        predictions = self.model_fold1.predict(X1)

        if self.mode.lower() == 'max_vote':
            predictions = np.round(predictions)
            unique_elements = np.unique(predictions)
            count_array = np.array([sum(predictions == unique_element)
                                    for unique_element in unique_elements])
            pred = np.argmax(count_array) if len(
                count_array) > 1 else unique_elements[0]
            conf = 1 if len(
                count_array) == 1 else count_array[pred]*1.0/np.sum(count_array)
        elif self.mode.lower() == 'avg':
            conf = np.mean(predictions)
            pred = np.round(conf)

        pred_str = 'faced' if pred == 1 else 'defaced'
        conf = conf if pred == 1 else 1.0 - conf

        if conf < self.threshold:
            print(
                f"Confidence: {conf} < threshold: {self.threshold} Re-evaluating on the entire volume")

            _X = self.inference_generator.get_data(vol, all=True)

            predictions_all_f1 = self.model_fold1.predict(_X)
            predictions_all_f2 = self.model_fold2.predict(_X)
            predictions_all_f3 = self.model_fold3.predict(_X)

            predictions_all = np.squeeze(np.concatenate([predictions_all_f1,
                                                         predictions_all_f2,
                                                         predictions_all_f3],
                                                        axis=0))
            conf = np.mean(predictions_all)
            pred = np.round(conf)
            pred_str = 'faced' if pred == 1 else 'defaced'
            conf = conf if pred == 1 else 1.0 - conf

        print("Given volume is " + pred_str +
                  " with confidence of: {}".format(conf))
        return pred, conf
