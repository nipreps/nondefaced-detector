import numpy as np
import os
import nibabel as nib
import random
import time
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


def relu6(x):
    """Custom activation using relu6"""
    return K.relu(x, max_value=6)


def ConvBNrelu(x, filters=32, kernel=3, strides=1, padding='same'):
    """
    """
    x = layers.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    return x


def TruncatedSubmodel(input_layer):
    """
    """
    conv1 = ConvBNrelu(input_layer, filters=8, kernel=3, strides=1, padding='same')
    conv1 = ConvBNrelu(conv1, filters=8, kernel=3, strides=1, padding='same')
    conv1 = layers.MaxPooling2D()(conv1)

    conv2 = ConvBNrelu(conv1, filters=16, kernel=3, strides=1, padding='same')
    conv2 = ConvBNrelu(conv2, filters=16, kernel=3, strides=1, padding='same')
    conv2 = layers.MaxPooling2D()(conv2)

    conv3 = ConvBNrelu(conv2, filters=32, kernel=3, strides=1, padding='same')
    conv3 = ConvBNrelu(conv3, filters=32, kernel=3, strides=1, padding='same')
    conv3 = layers.MaxPooling2D()(conv3)

    out = layers.Flatten()(conv3)
    return out

def ClassifierHead(layer, dropout):
    """
    """
    out = layers.Dense(256, activation='relu')(layer)
    out = layers.Dropout(dropout)(out)
    out = layers.Dense(1, activation='sigmoid', name='output_node')(out)
    return out

def Submodel(input_shape = (32, 32), dropout = 0.4, name = 'axial'):
    """
    """
    input_layer = layers.Input(shape=input_shape + (1,), name=name)
    features = TruncatedSubmodel(input_layer)
    prob = ClassifierHead(features, dropout)
    return models.Model(input_layer, prob)


def CombinedClassifier(axial_wts, coronal_wts, sagittal_wts, 
                    input_shape=(32, 32), dropout=0.4):
    """
    """
    
    axial_layer = layers.Input(shape=input_shape + (1,), name='axial')
    axial_features = TruncatedSubmodel(axial_layer)

    coronal_layer = layers.Input(shape=input_shape + (1,), name='coronal')
    coronal_features = TruncatedSubmodel(coronal_layer)
    
    sagittal_layer = layers.Input(shape=input_shape + (1,), name='sagittal')
    sagittal_features = TruncatedSubmodel(sagittal_layer)


    merge_features = [axial_features, sagittal_features, coronal_features]
    add_features = layers.Add()(merge_features)
    prob = ClassifierHead(add_features, dropout)

    model = models.Model(inputs=[axial_layer, coronal_layer, sagittal_layer], 
                                                outputs=prob)
    return model

if __name__ == '__main__':
    axial = Submodel(name='axial')
    coronal = Submodel(name='coronal')
    sagittal = Submodel(name='sagittal')

    combined = CombinedClassifier('a', 'b', 'c')
