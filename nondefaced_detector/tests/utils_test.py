import os
import pytest

import datalad.api

from nondefaced_detector.utils import get_datalad


def test_utils():

    get_datalad()
    cache_dir = '/tmp/nondefaced-detector-reproducibility'
    assert(os.path.exists(cache_dir))
    assert(os.path.exists(os.path.join(cache_dir, 'pretrained_weights')))
    datalad.api.ls(cache_dir, long_=True)
    assert(datalad.api.remove(cache_dir))
