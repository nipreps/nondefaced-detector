import sys, os
from nondefaced_detector.models.modelN import CombinedClassifier
from nondefaced_detector.dataloaders.dataset import get_dataset


# Tf packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras import losses

def inference(tfrecords_path, weights_path, wts_root):
    
    model = CombinedClassifier(
        input_shape=(128, 128), dropout=0.4, wts_root=wts_root, trainable=False)
    
    model.load_weights(os.path.abspath(weights_path))
    model.trainable = False
    
    dataset_test = get_dataset(
        file_pattern=os.path.join(tfrecords_path, "data-test_*"),
        n_classes=2,
        batch_size=16,
        volume_shape=(128, 128, 128),
        plane='combined',
        mode='test'
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
    predictions = (model.predict(dataset_test) > 0.5).astype(int)
    
    
    return predictions

if __name__ == "__main__":
    ROOTDIR = '/tf/shank/HDDLinux/Stanford/data/mriqc-shared/test_ixi'
    tfrecords_path = os.path.join(ROOTDIR, "tfrecords")
    weights_path = '/tf/shank/HDDLinux/Stanford/data/mriqc-shared/experiments/experiment_B/128/model_save_dir_full/weights/combined/best-wts.h5'
    wts_root = '/tf/shank/HDDLinux/Stanford/data/mriqc-shared/experiments/experiment_B/128/model_save_dir_full/weights'
    inference(tfrecords_path, weights_path, wts_root)
