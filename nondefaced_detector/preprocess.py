"""Script to preprocess volumes"""

import functools
import sys, os
import tempfile

import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm


from nondefaced_detector.preprocessing.conform       import conform_data
from nondefaced_detector.helpers                     import utils
from nondefaced_detector.preprocessing.normalization import clip, normalize, standardize


def preprocess(
    vol_path,
    conform_volume_to=(128, 128, 128),
    conform_zooms=(2.0, 2.0, 2.0),
    save_path=None,
    with_label=False,
):

    try:
        vpath = vol_path
        if with_label:
            if len(vol_path) != 2:
                raise ValueError(
                    "The vol_path must have length of 2 when with_label=True"
                )

            vpath, label = vol_path

        spath = os.path.join(os.path.dirname(vpath), "preprocessed")
        if save_path:
            spath = os.path.join(save_path, "preprocessed")

        os.makedirs(spath, exist_ok=True)

        volume, affine, _ = utils.load_vol(vpath)

        # Prepocessing
        volume = clip(volume, q=90)
        volume = standardize(volume)
        volume = normalize(volume)

        tmp_preprocess_vol = tempfile.NamedTemporaryFile(
            suffix=".nii.gz",
            delete=True,
            dir=spath,
        )

        utils.save_vol(tmp_preprocess_vol.name, volume, affine)

        tmp_conform_vol = os.path.join(spath, os.path.basename(vpath))

        conform_data(
            tmp_preprocess_vol.name,
            out_file=tmp_conform_vol,
            out_size=conform_volume_to,
            out_zooms=conform_zooms,
        )

        tmp_preprocess_vol.close()

        if with_label:
            return (tmp_conform_vol, label)
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
    with_label=True,
):

    try:
        if with_label:
            for pair in volume_filepaths:
                if len(pair) != 2:
                    raise ValueError(
                        "all items in 'volume_filepaths' must have length of 2, but"
                        " found at least one item with lenght != 2."
                    )

        map_fn = functools.partial(
            preprocess,
            conform_volume_to=conform_volume_to,
            conform_zooms=conform_zooms,
            save_path=save_path,
            with_label=with_label,
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
