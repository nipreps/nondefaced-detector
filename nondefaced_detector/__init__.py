from distutils.version import LooseVersion

import tensorflow as tf

import nondefaced_detector.dataloaders
import nondefaced_detector.helpers
import nondefaced_detector.prediction
import nondefaced_detector.preprocess
import nondefaced_detector.preprocessing
import nondefaced_detector.training

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    raise ValueError(
        "tensorflow>=2.0.0 must be installed but found version {}".format(
            tf.__version__
        )
    )
del LooseVersion, tf
