API Reference
==============

.. _base_ref:

:mod:`nondefaced_detector.models`: Model functions
----------------------------------------------------------------

.. autosummary:: nondefaced_detector.models
   :toctree: generated/

   nondefaced_detector.models.model.ConvBNrelu
   nondefaced_detector.models.model.TruncatedSubmodel
   nondefaced_detector.models.model.ClassifierHead
   nondefaced_detector.models.model.Submodel
   nondefaced_detector.models.model.CombinedClassifier

.. currentmodule:: nondefaced_detector

.. _calibration_ref:

:mod:`nondefaced_detector.dataloaders`: Dataset functions
----------------------------------------------------------------

.. autosummary:: nondefaced_detector.dataloaders
   :toctree: generated/

   nondefaced_detector.dataloaders.dataset.get_dataset

.. currentmodule:: nondefaced_detector

.. _calibration_ref:


:mod:`nondefaced_detector.preprocess`: Preprocess input volumes
----------------------------------------------------------------

.. autosummary:: nondefaced_detector.preprocess
   :toctree: generated/

   nondefaced_detector.preprocess.preprocess
   nondefaced_detector.preprocess.preprocess_parallel

.. currentmodule:: nondefaced_detector

.. _calibration_ref:

:mod:`nondefaced_detector.preprocessing`: Helper functions for the preprocess module
-------------------------------------------------------------------------------------

.. autosummary:: nondefaced_detector.preprocessing
   :toctree: generated/

   nondefaced_detector.preprocessing.conform.conform_data
   nondefaced_detector.preprocessing.normalization.clip
   nondefaced_detector.preprocessing.normalization.standardize
   nondefaced_detector.preprocessing.normalization.normalize

.. currentmodule:: nondefaced_detector

.. _calibration_ref:

:mod:`nondefaced_detector.training`: Training
------------------------------------------------

.. autosummary:: nondefaced_detector.training
   :toctree: generated/

   nondefaced_detector.training.training.train

.. currentmodule:: nondefaced_detector

.. _calibration_ref:


:mod:`nondefaced_detector.prediction`: Making predictions
----------------------------------------------------------

.. autosummary:: nondefaced_detector.prediction
   :toctree: generated/

   nondefaced_detector.prediction.predict
   nondefaced_detector.prediction._structural_slice
   nondefaced_detector.prediction._get_model

.. currentmodule:: nondefaced_detector

.. _calibration_ref:

:mod:`nondefaced_detector.inference`: Inference
------------------------------------------------

.. autosummary:: nondefaced_detector.inference
   :toctree: generated/

   nondefaced_detector.inference.inference

.. currentmodule:: nondefaced_detector

.. _calibration_ref:

:mod:`nondefaced_detector.helpers`: Helper functions
-----------------------------------------------------

.. autosummary:: nondefaced_detector.helpers
   :toctree: generated/

   nondefaced_detector.helpers.utils.is_gz_file
   nondefaced_detector.helpers.utils.save_vol
   nondefaced_detector.helpers.utils.load_vol
   nondefaced_detector.helpers.utils.imshow
   nondefaced_detector.helpers.utils.get_available_gpu

.. currentmodule:: nondefaced_detector

.. _calibration_ref:

:mod:`nondefaced_detector.utils`: Utility functions
-----------------------------------------------------

.. autosummary:: nondefaced_detector.utils
   :toctree: generated/

   nondefaced_detector.utils.get_datalad

.. currentmodule:: nondefaced_detector

.. _calibration_ref:
