"""Script to preprocess volumes"""

import functools
import os
import tempfile

import multiprocessing as mp
import tensorflow as tf

from tqdm import tqdm

from nondefaced_detector.preprocessing.conform import conform_data
from nondefaced_detector.helpers import utils
from nondefaced_detector.preprocessing.normalization import clip, normalize, standardize

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess(
    vol_path,
    conform_volume_to=(128, 128, 128),
    conform_zooms=(2.0, 2.0, 2.0),
    save_path=None,
    with_label=False,
):
    """Preprocess input volumes before prediction.

    Parameters
    ----------
    vol_path : str - Path or tuple of length 2 (str - Path, int)
        The path to the input volume. If the `with_label` flag is True, the
        vol_path is required to be a tuple of size 2 - (vol_path, label)
    conform_volume_to : tuple of length 3, optional, default=(128 128, 128)
        The shape the volume will be conformed to. Note: The pretrained model
        was trained using the conform size of (128, 128, 128) and assumes the
        volume shape as such.
    save_path : str - Path, optional
        The path where the output volume is saved. If none is provided, the
        output volume will be saved under `vol_path/preprocessed`
    with_label: bool, optional
        If True, the input vol_path is required to be a tuple of 2
        (vol_path, label)

    Returns
    -------
    str - Path
        Path to the where the preprocessed volume is stored.
        (Path, label) if with_label is True.
    """

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
            suffix=".nii.gz", delete=True, dir=spath
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
    num_parallel_calls=AUTOTUNE,
    conform_volume_to=(128, 128, 128),
    conform_zooms=(2.0, 2.0, 2.0),
    save_path=None,
    with_label=True,
):
    """Preprocess multiple input volumes before prediction in parallel.

    Parameters
    ----------
    volume_filepaths: list of str - Path or list of tuple of length 2 [(str - Path, int), ...]
        A list of paths to the input volumes. If the `with_label` flag is True, the
        volume_filepaths is required to be a list of tuples of size 2 - (volume_filepath, label)
    num_parallel_calls: int
        Number of parallel calls to make for preprocessing.
    conform_volume_to: tuple of length 3, optional, default=(128 128, 128)
        The shape the volume will be conformed to. Note: The pretrained model
        was trained using the conform size of (128, 128, 128) and assumes the
        volume shape as such.
    conform_zooms: tuple of size 3, optional, default=(2.0, 2.0, 2.0)
        The zoom of the resampled output.
    save_path: str - Path, optional
        The path where the output volume is saved. If none is provided, the
        output volume will be saved under `volume_filepath/preprocessed`
    with_label: bool, optional
        If True, each volume_filepath is required to be a tuple of 2 (volume_filepath, label)

    Returns
    -------
    list of str
        List of str paths to the where each preprocessed volume is stored.
        [(Path, label), ...] if with_label is True.
    """

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
    """ Function to remove temp files created during preprocessing."""
    for p in args:
        if os.path.exists(p):
            os.remove(p)
