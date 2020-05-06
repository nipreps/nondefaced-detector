# Std packages
import sys, os
import glob

# sys.path.append("..")

# Custom packages
from ..models import modelN
from ..dataloaders.dataset import get_dataset

# Tf packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
import nobrainer
from tensorflow.keras import metrics
from tensorflow.keras import losses

ROOTDIR = "/work/06850/sbansal6/maverick2/mriqc-shared/"


def scheduler(epoch):
    if epoch < 3:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def train(
    volume_shape=(64, 64, 64),
    image_size=(64, 64),
    dropout=0.2,
    batch_size=8,
    n_classes=2,
    n_epochs=30,
    fold=1
):
    tpaths = glob.glob(ROOTDIR+"tfrecords/tfrecords_fold_{}/data-train_*".format(fold))
    vpaths = glob.glob(ROOTDIR+"tfrecords/tfrecords_fold_{}/data-valid_*".format(fold))

    planes = ["axial", "coronal", "sagittal", "combined"]

    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = batch_size
    global_batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    model_save_path = "./model_save_dir/folds_{}".format(fold)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    cp_save_path = os.path.join(model_save_path, "weights")

    logdir_path = os.path.join(model_save_path, "tb_logs")
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)

    for plane in planes:

        logdir = os.path.join(logdir_path, plane)
        os.makedirs(logdir, exist_ok=True)

        tbCallback = TensorBoard(
            log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
        )

        os.makedirs(os.path.join(cp_save_path, plane), exist_ok=True)

        model_checkpoint = ModelCheckpoint(
            os.path.join(cp_save_path, plane, "best-wts.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )

        with strategy.scope():

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
            print(model.summary())

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
            ROOTDIR + "tfrecords/tfrecords_fold_{}/data-train_*".format(fold),
            n_classes=n_classes,
            batch_size=global_batch_size,
            volume_shape=volume_shape,
            plane=plane,
            shuffle_buffer_size=global_batch_size,
        )

        dataset_valid = get_dataset(
            ROOTDIR + "tfrecords/tfrecords_fold_{}/data-valid_*".format(fold),
            n_classes=n_classes,
            batch_size=global_batch_size,
            volume_shape=volume_shape,
            plane=plane,
            shuffle_buffer_size=global_batch_size,
        )

        steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
            n_volumes=len(tpaths),
            volume_shape=volume_shape,
            block_shape=volume_shape,
            batch_size=global_batch_size,
        )

        validation_steps = nobrainer.dataset.get_steps_per_epoch(
            n_volumes=len(vpaths),
            volume_shape=volume_shape,
            block_shape=volume_shape,
            batch_size=global_batch_size,
        )

        print(steps_per_epoch, validation_steps)

        lrcallback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        model.fit(
            dataset_train,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=dataset_valid,
            validation_steps=validation_steps,
            callbacks=[tbCallback, model_checkpoint],
        )

        del model
        K.clear_session()


if __name__ == "__main__":
    for fold in range(1, 11):
        train(fold=fold)
