"""Run inference on held-out test dataset."""

import argparse
import sys, os


# Tf packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam


from nondefaced_detector.models.modelN import CombinedClassifier
from nondefaced_detector.dataloaders.dataset import get_dataset


def inference(tfrecords_path, weights_path, wts_root):

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
    # predictions = (model.predict(dataset_test) > 0.5).astype(int)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("tfrecords", metavar="path", help="Path to tfrecords.")

    args = parser.parse_args()

    tfrecords_path = args.tfrecords
    weights_path = "models/pretrained_weights/combined/best-wts.h5"
    wts_root = "models/pretrained_weights"
    inference(tfrecords_path, weights_path, wts_root)
