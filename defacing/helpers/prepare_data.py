import os, sys

sys.path.append("..")
import binascii
from helpers.utils import load_vol, save_vol
from preprocessing.normalization import standardize_volume, normalize_volume
from preprocessing.conform import conform_data
import numpy as np
import nibabel as nb
from glob import glob
from pathlib import Path
from shutil import *
import subprocess


orig_data_face = "/work/01329/poldrack/data/mriqc-net/data/face/T1w"
orig_data_deface = "/work/01329/poldrack/data/mriqc-net/data/defaced"

save_data_face = "/work/06850/sbansal6/maverick2/mriqc-shared/conformed/face"
save_data_deface = "/work/06850/sbansal6/maverick2/mriqc-shared/conformed/deface"

os.makedirs(save_data_face, exist_ok=True)
os.makedirs(save_data_deface, exist_ok=True)


conform_size = (64, 64, 64)


def is_gz_file(filepath):
    if os.path.splitext(filepath)[1] == ".gz":
        with open(filepath, "rb") as test_f:
            return binascii.hexlify(test_f.read(2)) == b"1f8b"
    return False


def preprocess(pth, conform_size, save_data_path):
    """
    """
    filename = pth.split("/")[-1]
    print("Confirmation step")
    volume = conform_data(pth, out_size=conform_size)

#     print("Normalize/Standardize step")
#     volume = normalize_volume(standardize_volume(volume))
#     save_path = os.path.join(save_data_path, filename)

#     newaffine = np.eye(4)
#     newaffine[:3, 3] = -0.5 * (np.array(conform_size) - 1)
#     nii = nb.Nifti1Image(volume, newaffine, None)

#     print("Save new affine")
#     nii.to_filename(save_path)
#     return save_path


def checkNonConformed(orig_path, save_path):

    conform = []
    orig = []

    for path in glob(save_path + "/*/*.nii*"):
        tempname = path.split("/")[-1]
        ds = path.split("/")[-2]
        conform.append(ds + "/" + tempname)

    print("Total Conformed: ", len(conform))

    for path in glob(orig_path + "/*/*.nii*"):
        tempname = path.split("/")[-1]
        ds = path.split("/")[-2]
        orig.append(ds + "/" + tempname)

    print("Total Original: ", len(orig))

    print("Total not conformed: ", len(orig) - len(conform))

    count = 0
    for f in orig:
        exists = False
        for fc in conform:
            if fc in f:
                exists = True
        if not exists:
            count += 1
            print("Not conformed file: ", f)


for path in glob(orig_data_face + "/*/*.nii*"):
    try:
        print("Orig Path: ", path)
        if not is_gz_file(path) and os.path.splitext(path)[1] == ".gz":
            tempname = path.split("/")[-1]
            ds = path.split("/")[-2]
            rename_file = os.path.splitext(tempname)[0]
            dst = os.path.join(save_data_face, rename_file)
            print(dst)
            subprocess.call(["cp", path, dst])
            ds_save_path = os.path.join(save_data_face, ds)
            if not os.path.exists(ds_save_path):
                os.makedirs(ds_save_path)
            print(preprocess(dst, conform_size, save_data_path=ds_save_path))
        else:
            ds = path.split("/")[-2]
            ds_save_path = os.path.join(save_data_face, ds)
            if not os.path.exists(ds_save_path):
                os.makedirs(ds_save_path)
            print(preprocess(path, conform_size, save_data_path=ds_save_path))
    except:
        print("Preprocessing incomplete. Exception occurred.")
        pass


for path in glob(orig_data_deface + "/*/*.nii*"):
    try:
        print("Orig Path: ", path)
        if not is_gz_file(path) and os.path.splitext(path)[1] == ".gz":
            tempname = path.split("/")[-1]
            ds = path.split("/")[-2]
            rename_file = os.path.splitext(tempname)[0]
            dst = os.path.join(save_data_deface, rename_file)
            print(dst)
            subprocess.call(["cp", path, dst])
            ds_save_path = os.path.join(save_data_deface, ds)
            if not os.path.exists(ds_save_path):
                os.makedirs(ds_save_path)
            print(preprocess(dst, conform_size, save_data_path=ds_save_path))
        else:
            ds = path.split("/")[-2]
            ds_save_path = os.path.join(save_data_deface, ds)
            if not os.path.exists(ds_save_path):
                os.makedirs(ds_save_path)
            print(preprocess(path, conform_size, save_data_path=ds_save_path))
    except:
        print("Preprocessing incomplete. Exception occurred.")
        pass
    

checkNonConformed(orig_data_face, save_data_face)
checkNonConformed(orig_data_deface, save_data_deface)
