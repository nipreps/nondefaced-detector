import nobrainer
from nobrainer.io import _is_gzipped
from nobrainer.volume import to_blocks
import tensorflow_probability as tfp
import tensorflow as tf
import glob
import numpy as np


def clip(x, q =90):
    """
    """
    min_val = 0
    max_val = tfp.stats.percentile(
                x, q, axis=None,
                preserve_gradients=False,
                name=None
                )
    x = tf.clip_by_value(
        x,
        min_val,
        max_val,
        name=None
        )
    return x

def standardize(x):
    """Standard score input tensor.
    Implements `(x - median(x)) / stdev(x)`.
    Parameters
    ----------
    x: tensor, values to standardize.
    Returns
    -------
    Tensor of standardized values. Output has mean 0 and standard deviation 1.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    median = tfp.stats.percentile(
                x, 50, axis=None,
                preserve_gradients=False,
                name=None
                )
    mean, var = tf.nn.moments(x, axes=None)
    std = tf.sqrt(var)
    return (x - median) / std


def normalize(x):
    """Normalize score input tensor.
    Implements `(x - mean(x)) / stdev(x)`.
    Parameters
    ----------
    x: tensor, values to standardize.
    Returns
    -------
    Tensor of standardized values. Output has mean 0 and standard deviation 1.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)

    max_value = tf.math.reduce_max(
                x,
                axis=None,
                keepdims=False, name=None
                )

    min_value = tf.math.reduce_min(
                x,
                axis=None,
                keepdims=False, name=None
                )
    return (x - min_value) / (max_value - min_value + 1e-3)