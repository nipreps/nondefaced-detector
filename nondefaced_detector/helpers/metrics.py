import os, sys
import glob
import random
import time
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import wget
from tensorflow.keras import losses
from tensorflow.python.client import device_lib
import GPUtil
from threading import Thread
import time


def class_0_acc(y_true, y_pred):
    """
    class 0 acc
    """
    y_pred = K.flatten(K.round(y_pred[..., 0]))
    y_true = K.flatten(K.round(y_true[..., 0]))

    nr = K.sum(y_pred * y_true)
    dr = K.sum(y_true)
    return nr / (dr + 1e-5)


def class_1_acc(y_true, y_pred):
    """
    class 1 acc
    """
    y_pred = K.flatten(K.round(y_pred[..., 1]))
    y_true = K.flatten(K.round(y_true[..., 1]))

    nr = K.sum(y_pred * y_true)
    dr = K.sum(y_true)
    return nr / (dr + 1e-5)


def roc_auc_score(y_true, y_pred):
    """ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))


# Defining customized metrics
def sensitivity(y_true, y_pred):
    """Sensitivity = True Positives / (True Positives + False Negatives)"""

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (all_positives + K.epsilon())


def specificity(y_true, y_pred):
    """Specificity = True Negatives / (True Negatives + False Positives)"""

    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    all_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (all_negatives + K.epsilon())
