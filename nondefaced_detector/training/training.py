# Std packages
import sys, os
import glob
import math

sys.path.append("../..")

# Custom packages
import defacing
from defacing.models import modelN
from defacing.dataloaders.dataset import get_dataset

# Tf packages
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
    EarlyStopping,
)
from tensorflow.keras import metrics
from tensorflow.keras import losses


def scheduler(epoch):
    if epoch < 3:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def train(
    csv_path,
    model_save_path,
    tfrecords_path,
    volume_shape=(64, 64, 64),
    image_size=(64, 64),
    dropout=0.2,
    batch_size=16,
    n_classes=2,
    n_epochs=15,
    percent=100,
    mode="CV",
):

    train_csv_path = os.path.join(csv_path, "training.csv")
    train_paths = pd.read_csv(train_csv_path)["X"].values
    train_labels = pd.read_csv(train_csv_path)["Y"].values

    if mode == "CV":
        valid_csv_path = os.path.join(csv_path, "validation.csv")
        valid_paths = pd.read_csv(valid_csv_path)["X"].values
        valid_labels = pd.read_csv(valid_csv_path)["Y"].values

    weights = class_weight.compute_class_weight(
        "balanced", np.unique(train_labels), train_labels
    )
    weights = dict(enumerate(weights))

    print(weights)

    planes = ["axial", "coronal", "sagittal", "combined"]

    global_batch_size = batch_size

    os.makedirs(model_save_path, exist_ok=True)
    cp_save_path = os.path.join(model_save_path, "weights")
    logdir_path = os.path.join(model_save_path, "tb_logs")
    metrics_path = os.path.join(model_save_path, "metrics")

    os.makedirs(metrics_path, exist_ok=True)
    #     os.makedirs(logdir_path, exist_ok=True)

    for plane in planes:

        logdir = os.path.join(logdir_path, plane)
        os.makedirs(logdir, exist_ok=True)

        tbCallback = TensorBoard(log_dir=logdir)

        os.makedirs(os.path.join(cp_save_path, plane), exist_ok=True)

        model_checkpoint = ModelCheckpoint(
            os.path.join(cp_save_path, plane, "best-wts.h5"),
            monitor="val_loss",
            save_weights_only=True,
            mode="min",
        )

        #         with strategy.scope():

        if not plane == "combined":
            lr = 1e-3
            model = modelN.Submodel(
                input_shape=image_size,
                dropout=dropout,
                name=plane,
                include_top=True,
                weights=None,
            )
        else:
            lr = 5e-4
            model = modelN.CombinedClassifier(
                input_shape=image_size,
                dropout=dropout,
                trainable=True,
                wts_root=cp_save_path,
            )

        print("Submodel: ", plane)
        #         print(model.summary())

        METRICS = [
            metrics.TruePositives(name="tp"),
            metrics.FalsePositives(name="fp"),
            metrics.TrueNegatives(name="tn"),
            metrics.FalseNegatives(name="fn"),
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ]

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=Adam(learning_rate=lr),
            metrics=METRICS,
        )

        print("GLOBAL BATCH SIZE: ", global_batch_size)

        dataset_train = get_dataset(
            file_pattern=os.path.join(tfrecords_path, "data-train_*"),
            n_classes=n_classes,
            batch_size=global_batch_size,
            volume_shape=volume_shape,
            plane=plane,
            shuffle_buffer_size=global_batch_size,
        )

        steps_per_epoch = math.ceil(len(train_paths) / batch_size)
        print(steps_per_epoch)

        # CALLBACKS
        lrcallback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if mode == "CV":
            earlystopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3
            )

            dataset_valid = get_dataset(
                file_pattern=os.path.join(tfrecords_path, "data-valid_*"),
                n_classes=n_classes,
                batch_size=global_batch_size,
                volume_shape=volume_shape,
                plane=plane,
                shuffle_buffer_size=global_batch_size,
            )

            validation_steps = math.ceil(len(valid_paths) / batch_size)

            history = model.fit(
                dataset_train,
                epochs=n_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=dataset_valid,
                validation_steps=validation_steps,
                callbacks=[tbCallback, model_checkpoint, earlystopping],
                class_weight=weights,
            )

            hist_df = pd.DataFrame(history.history)

        else:
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
            print(model.summary())
            print("Steps/Epoch: ", steps_per_epoch)
            history = model.fit(
                dataset_train,
                epochs=n_epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=[tbCallback, model_checkpoint, earlystopping],
                class_weight=weights,
            )

        hist_df = pd.DataFrame(history.history)
        jsonfile = os.path.join(metrics_path, plane + ".json")

        with open(jsonfile, mode="w") as f:
            hist_df.to_json(f)

        del model
        K.clear_session()

    return history


if __name__ == "__main__":
    ROOTDIR = (
        "/tf/shank/HDDLinux/Stanford/data/mriqc-shared/experiments/experiment_B/128"
    )
    csv_path = os.path.join(ROOTDIR, "csv_full")
    model_save_path = os.path.join(ROOTDIR, "model_save_dir_full")
    tfrecords_path = os.path.join(ROOTDIR, "tfrecords_full")

    history = train(
        csv_path,
        model_save_path,
        tfrecords_path,
        volume_shape=(128, 128, 128),
        image_size=(128, 128),
        mode="full",
    )
