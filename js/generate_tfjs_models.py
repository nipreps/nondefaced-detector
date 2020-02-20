import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflowjs as tfjs

sys.path.append('..')
from defacing.models.model import custom_model


model_fold1 = custom_model(input_shape=(64, 64),
                                        nclasses=2,
                                        multiencoders=True)
model_fold1.load_weights(os.path.abspath(
            '../defacing/saved_weights/best_cv1.h5'))
tfjs.converters.save_keras_model(model_fold1, 'models/model_fold1')


model_fold2 = custom_model(input_shape=(64, 64),
                                            nclasses=2,
                                            multiencoders=True)
model_fold2.load_weights(
                '../defacing/saved_weights/best_cv2.h5')

tfjs.converters.save_keras_model(model_fold2, 'models/model_fold2')


model_fold3 = custom_model(input_shape=(64, 64),
                                            nclasses=2,
                                            multiencoders=True)
model_fold3.load_weights(
                '../defacing/saved_weights/best_cv2.h5')
tfjs.converters.save_keras_model(model_fold3, 'models/model_fold3')
