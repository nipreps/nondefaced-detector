"""Main command-line interface for nondefaced-detector."""

import datetime
import json
import logging
import os
import platform
import sys

import click
import nibabel as nib
import numpy as np
import nobrainer
import tensorflow as tf

from nobrainer.io import read_csv as _read_csv

from nondefaced_detector            import __version__
from nondefaced_detector            import prediction
from nondefaced_detector.helpers    import utils
from nondefaced_detector.preprocess import preprocess as _preprocess


_option_kwds = {"show_default": True}


@click.group()
@click.version_option(__version__, message="%(prog)s version %(version)s")
def cli():
    """A framework to detect if a 3D MRI Volume has been defaced."""
    return


@cli.command()
@click.option(
    "-c", "--csv",
    type=click.Path(exists=True),
    required=True,
    **_option_kwds,
)
@click.option(
    "-p", "--preprocess-path",
    type=click.Path(exists=True),
    default=None,
    required=False,
    **_option_kwds,
)
@click.option(
    "-t",
    "--tfrecords-template",
    default="tfrecords/data_shard-{shard:03d}.tfrec",
    required=True,
    **_option_kwds,
)
@click.option(
    "-s", "--volume-shape",
    nargs=3,
    type=int,
    required=True,
    **_option_kwds
)
@click.option(
    "-n",
    "--examples-per-shard",
    type=int,
    default=100,
    help="Number of (feature, label) pairs per TFRecord file.",
    **_option_kwds,
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Print progress bar.",
    **_option_kwds
)
def convert(
    *
    csv,
    preprocess_path,
    tfrecords_template,
    volume_shape,
    examples_per_shard,
    verbose,
    
):
    """Preprocess MRI volumes and convert to Tfrecords. 
    
    NOTE: Volumes will all be the same shape after preprocessing.
    """
    volume_filepaths = _read_csv(csv)
    num_parallel_calls = None if num_parallel_calls == -1 else num_parallel_calls
    if num_parallel_calls is None:
        # Get number of processes allocated to the current process.
        # Note the difference from `os.cpu_count()`.
        num_parallel_calls = len(os.sched_getaffinity(0))

    return

@cli.command()
@click.argument("infile")
@click.argument("outfile")
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    default='../models/pretrained_weights',
    help="Path to model weights.",
    **_option_kwds,
)
@click.option(
    "-r",
    "--conform-volume-to",
    default=(128, 128, 128),
    type=int,
    nargs=3,
    help="Conform volume to this size before predicting.",
    **_option_kwds,
)
@click.option(
    "-p",
    "--preprocess-path",
    type=click.Path(exists=True),
    required=False,
    help="Path to save preprocessed volumes.",
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", 
    is_flag=True,
    help="Print progress bar.",
    **_option_kwds
)
def predict(
    *,
    infile,
    outfile,
    model,
    conform_volume_to,
    preprocess_path,
    verbose,
):
    """Predict labels from features using a trained model.

    The predictions are saved to OUTFILE.
    """

    if not verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel(logging.ERROR)

    if os.path.exists(outfile):
        raise FileExistsError(
            "Output file already exists. Will not overwrite {}".format(outfile)
        )


    ppath, cpath = _preprocess(infile)

    required_dirs = ['axial', 'coronal', 'sagittal', 'combined']

    for plane in required_dirs:
        if not os.path.isdir(os.path.join(model, plane)):
            raise ValueError('Missing {} directory in model path'.format(plane))

    volume, _, _ = utils.load_vol(cpath)
    predicted = prediction.predict(volume, model)


@cli.command()
def evaluate():
    """Evaluate a model's predictions against known labels."""
    click.echo(
        "Not implemented yet. In the future, this command will be used for evaluation."
    )
    sys.exit(-2)

@cli.command()
def info():
    """Return information about this system."""
    uname = platform.uname()
    s = f"""\
Python:
 Version: {platform.python_version()}
 Implementation: {platform.python_implementation()}
 64-bit: {sys.maxsize > 2**32}
 Packages:
  Nondefaced-Detector: {__version__}
  Nibabel: {nib.__version__}
  Numpy: {np.__version__}
  Nobrainer: {nobrainer.__version__}
  TensorFlow: {tf.__version__}
   GPU support: {tf.test.is_built_with_gpu_support()}
   GPU available: {bool(tf.config.list_physical_devices('GPU'))}
System:
 OSType: {uname.system}
 Release: {uname.release}
 Version: {uname.version}
 Architecture: {uname.machine}
Timestamp: {datetime.datetime.utcnow().strftime('%Y/%m/%d %T')}"""
    click.echo(s)

if __name__ == "__main__":
    cli()
