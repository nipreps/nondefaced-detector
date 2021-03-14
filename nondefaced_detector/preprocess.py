"""Script to preprocess volumes"""

import functools
import sys, os
import tempfile

import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm

from nondefaced_detector.preprocessing.normalization import clip, normalize, standardize
from nondefaced_detector.preprocessing.conform import conform_data
from nondefaced_detector.helpers import utils


def preprocess(
    vol_path,
    conform_volume_to=(128, 128, 128),
    conform_zooms=(2.0, 2.0, 2.0),
    save_path=None,
):

    try:
        if not save_path:
            save_path = os.path.join(os.path.dirname(vol_path), "preprocessed")
            os.makedirs(save_path, exist_ok=True)

        volume, affine, _ = utils.load_vol(vol_path)

        # Prepocessing
        volume = clip(volume, q=90)
        volume = normalize(volume)
        volume = standardize(volume)

        tmp_preprocess_vol = tempfile.NamedTemporaryFile(
            suffix=".nii.gz",
            delete=True,
            dir=save_path,
        )

        utils.save_vol(tmp_preprocess_vol.name, volume, affine)

        tmp_conform_vol = os.path.join(save_path, os.path.basename(vol_path))

        conform_data(
            tmp_preprocess_vol.name,
            out_file=tmp_conform_vol,
            out_size=conform_volume_to,
            out_zooms=conform_zooms,
        )

        tmp_preprocess_vol.close()

        return tmp_conform_vol

    except Exception as e:
        print(e)
        return


def preprocess_parallel(
    volume_filepaths,
    num_parallel_calls=None,
    conform_volume_to=(128, 128, 128),
    conform_zooms=(2.0, 2.0, 2.0),
    save_path=None,
):

    try:
        map_fn = functools.partial(
            preprocess,
            conform_volume_to=conform_volume_to,
            conform_zooms=conform_zooms,
            save_path=save_path,
        )

        if num_parallel_calls is None:
            # Get number of eligible CPUs.
            num_parallel_calls = len(os.sched_getaffinity(0))

        print("Preprocessing {} examples".format(len(volume_filepaths)))

        outputs = []

        if num_parallel_calls == 1:
            for vf in tqdm(volume_filepaths, total=len(volume_filepaths)):
                result = map_fn(vf)
                outputs.append(result)
        else:
            pool = mp.Pool(num_parallel_calls)
            for result in tqdm(
                pool.imap(func=map_fn, iterable=volume_filepaths),
                total=len(volume_filepaths),
            ):
                outputs.append(result)

        return outputs

    except Exception as e:
        print(e)
        return


def cleanup_files(*args):
    for p in args:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    vol_path = "../examples/sample_vols/faced/example1.nii.gz"
    cpath = preprocess(vol_path)
    print(cpath)
    cleanup_files(cpath)

    # parallel test
