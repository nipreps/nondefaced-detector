from distutils.version import LooseVersion
from pkg_resources import get_distribution, DistributionNotFound

import tensorflow as tf

import nondefaced_detector.dataloaders
import nondefaced_detector.helpers
import nondefaced_detector.prediction
import nondefaced_detector.preprocess
import nondefaced_detector.preprocessing
import nondefaced_detector.utils

try:
    __version__ = get_distribution("nondefaced-detector").version
except DistributionNotFound:
    # package is not installed
    raise ValueError("nondefaced-detector must be installed")

if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    raise ValueError(
        "tensorflow>=2.0.0 must be installed but found version {}".format(
            tf.__version__
        )
    )
del LooseVersion, tf
