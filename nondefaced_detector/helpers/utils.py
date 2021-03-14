"""Helper functions for nondefaced-detector."""


import binascii
import glob
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib


def is_gz_file(filepath):
    if os.path.splitext(filepath)[1] == ".gz":
        with open(filepath, "rb") as test_f:
            return binascii.hexlify(test_f.read(2)) == b"1f8b"
    return False


def save_vol(save_path, tensor_3d, affine):
    """
    save_path: path to write the volume to
    tensor_3d: 3D volume which needs to be saved
    affine: image orientation, translation
    """
    directory = os.path.dirname(save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    volume = nib.Nifti1Image(tensor_3d, affine)
    volume.set_data_dtype(np.float32)
    nib.save(volume, save_path)


def load_vol(load_path):
    """
    load_path: volume path to load
    return:
            volume: loaded 3D volume
            affine: affine data specific to the volume
    """
    if not os.path.exists(load_path):
        raise ValueError("path doesn't exist")

    nib_vol = nib.load(load_path)
    vol_data = nib_vol.get_data()
    vol_affine = nib_vol.affine

    return np.array(vol_data), vol_affine, vol_data.shape


def grid_to_single(image_batch, label_image=False):
    shape = image_batch.shape
    # print (shape)
    n = int(np.sqrt(shape[0]))
    x = shape[1]
    y = shape[2]
    if not label_image:
        img_array = np.zeros((x * n, y * n, 3))
    else:
        img_array = np.zeros((x * n, y * n))

    # print (img_array.shape)
    idx = 0
    for i in range(n):
        for j in range(n):
            # print (i*x, (i+1)*x, j*y, (j+1)*y, idx)
            # imshow(image_batch[idx])
            if not label_image:
                img_array[i * x : (i + 1) * x, j * y : (j + 1) * y, :] = image_batch[
                    idx
                ]
            else:
                img_array[i * x : (i + 1) * x, j * y : (j + 1) * y] = image_batch[idx]
            idx = idx + 1
    return img_array


def imshow(*args, **kwargs):
    """Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues'])
    """
    cmap = kwargs.get("cmap", "gray")
    title = kwargs.get("title", "")
    axis_off = kwargs.get("axis_off", "")
    if len(args) == 0:
        raise ValueError("No images given to imshow")
    elif len(args) == 1:
        plt.title(title)
        plt.imshow(args[0], interpolation="none")
    else:
        n = len(args)
        if type(cmap) == str:
            cmap = [cmap] * n
        if type(title) == str:
            title = [title] * n
        plt.figure(figsize=(n * 5, 10))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
            if axis_off:
                plt.axis("off")
    plt.show()


def performance_evaluator(gt, prediction):
    """"""

    gt = np.array(gt, dtype="float32")
    prediction = np.array(prediction, dtype="float32")
    correct = np.sum(prediction == gt)
    accuracy = correct * 100.0 / (len(gt) + 1e-5)

    cm = confusion_matrix(gt, prediction)

    tp = sum((prediction == 1) * (gt == 1))
    tn = sum((prediction == 0) * (gt == 0))
    fp = sum((prediction == 1) * (gt == 0))
    fn = sum((prediction == 0) * (gt == 1))

    sensitivity = tp * 1.0 / (tp + fn + 1e-3)
    specificity = tn * 1.0 / (tn + fp + 1e-3)

    print("Inference Logs =======================")
    print("confusion matrix", cm)
    print("Accuracy on Inference Data: {:.3f}".format(accuracy))
    print("Sensitivity on Inference Data: {:.3f}".format(sensitivity))
    print("Specificity on Inference Data: {:.3f}".format(specificity))

    json = {
        "true_positive": str(tp),
        "false_positive": str(fp),
        "true_negative": str(tn),
        "false_negative": str(fn),
        "specificity": str(specificity),
        "sensitivity": str(sensitivity),
        "accuracy": str(accuracy),
    }
    return json


def get_available_gpus():
    """
    Get the total number of GPUs available
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:

            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]


if __name__ == "__main__":
    vol, aff, _ = load_vol(
        "/home/pi/mri-face-detector/sample_vols/faced/example4.nii.gz"
    )
    vol = resize_sitk(vol, outputSize=(64, 64, 64))
    save_vol(
        "/home/pi/mri-face-detector/sample_vols/faced/example4_conf.nii.gz", vol, aff
    )
