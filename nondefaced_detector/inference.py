"""Standalone inference script for held-out test dataset."""

import argparse
import os

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam

from nondefaced_detector.models.model import CombinedClassifier
from nondefaced_detector.dataloaders.dataset import get_dataset


def inference(tfrecords_path, weights_path, wts_root):
    """ Inference function to reproduce original model scores. This
    script can be run as a standalone using python inference.py.
    For more information try: `python inference.py -h`

    Parameters
    ----------
    tfrecords_path: str
        The path to directory containing preprocessed tfrecords.
    weights_path: str
        The path to the combined model weights. A copy of the
        weights can be found here:
        https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility/src/master/pretrained_weights/combined
    wts_root: str
        The path to the root directory of all the model weights.
        A copy of the weights can be found here:
        https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility/src/master/pretrained_weights
    """

    model = CombinedClassifier(
        input_shape=(128, 128), dropout=0.4, wts_root=wts_root, trainable=False
    )

    model.load_weights(os.path.abspath(weights_path))
    model.trainable = False

    dataset_test = get_dataset(
        file_pattern=os.path.join(tfrecords_path, "data-test_*"),
        n_classes=2,
        batch_size=16,
        volume_shape=(128, 128, 128),
        plane="combined",
        mode="test",
    )

    METRICS = [
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
        metrics.AUC(name="auc"),
    ]

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=Adam(learning_rate=1e-3),
        metrics=METRICS,
    )

    model.evaluate(dataset_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tfrecords", metavar="tfrecords_path", help="Path to tfrecords."
    )
    parser.add_argument(
        "model_path", metavar="model_path", help="Path to pretrained model weights."
    )

    args = parser.parse_args()

    tfrecords_path = args.tfrecords
    model_path = args.model_path
    combined_path = os.path.join(model_path, "combined/best-wts.h5")
    inference(tfrecords_path, combined_path, model_path)
