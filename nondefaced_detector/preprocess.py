"""Script to preprocess volumes"""

import tempfile
import sys, os

from nondefaced_detector.preprocessing.normalization import clip, normalize, standardize
from nondefaced_detector.preprocessing.conform       import conform_data
from nondefaced_detector.helpers                     import utils


def preprocess(
    vol_path,
    conform_volume_to=(128, 128, 128),
    conform_zooms=(2.0, 2.0, 2.0),
    save_path='/tmp',
):

    try:
        volume, affine, _ = utils.load_vol(vol_path)

        # Prepocessing
        volume = clip(volume, q=90)
        volume = normalize(volume)
        volume = standardize(volume)

        tmp_preprocess_vol = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False, dir=save_path).name
        utils.save_vol(tmp_preprocess_vol, volume, affine)

        tmp_conform_vol = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False, dir=save_path).name

        conform_data(
            tmp_preprocess_vol,
            out_file=tmp_conform_vol,
            out_size=conform_volume_to,
            out_zooms=conform_zooms)
        
        return tmp_preprocess_vol, tmp_conform_vol
    except Exception as e:
        return


def cleanup_files(*args):
    print("Cleaning up...")
    for p in args:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    vol_path = "../examples/sample_vols/IXI002-Guys-0828-T1.nii.gz"
    ppath, cpath = preprocess(vol_path)
    print(ppath, cpath)
    cleanup_files(ppath, cpath)
