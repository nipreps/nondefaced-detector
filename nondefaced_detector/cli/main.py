"""Main command-line interface for nondefaced-detector."""

import click
import csv
import datetime
import errno
import json
import logging
import os
import platform
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import nibabel as nib
import numpy as np
import nobrainer

# import tensorflow as tf

from nobrainer.io       import read_csv
from nobrainer.io       import verify_features_labels
from nobrainer.tfrecord import write as _write_tfrecord

from nondefaced_detector import __version__
from nondefaced_detector import prediction

from nondefaced_detector.helpers    import utils
from nondefaced_detector.preprocess import preprocess, cleanup_files
from nondefaced_detector.preprocess import preprocess_parallel


_option_kwds = {"show_default": True}


@click.group()
@click.version_option(__version__, message="%(prog)s version %(version)s")
def cli():
    """A framework to detect if a 3D MRI Volume has been defaced."""
    return


@cli.command()
@click.option(
    "-c",
    "--csv",
    type=click.Path(exists=True),
    required=True,
    **_option_kwds,
)
@click.option(
    "-p",
    "--preprocess-path",
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
    "-s",
    "--volume-shape",
    default=(128, 128, 128),
    nargs=3,
    type=int,
    required=True,
    **_option_kwds,
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
    "-j",
    "--num-parallel-calls",
    default=-1,
    type=int,
    help="Number of processes to use. If -1, uses all available processes.",
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress bar.", **_option_kwds
)
def convert(
    csv,
    preprocess_path,
    tfrecords_template,
    volume_shape,
    examples_per_shard,
    num_parallel_calls,
    verbose,
):
    """Preprocess MRI volumes and convert to Tfrecords.

    NOTE: Volumes will all be the same shape after preprocessing.
    """

    volume_filepaths = read_csv(csv)

    num_parallel_calls = None if num_parallel_calls == -1 else num_parallel_calls
    if num_parallel_calls is None:
        # Get number of processes allocated to the current process.
        # Note the difference from `os.cpu_count()`.
        num_parallel_calls = len(os.sched_getaffinity(0))

    invalid_pairs = verify_features_labels(
        volume_filepaths,
        check_labels_int=True,
        num_parallel_calls=num_parallel_calls,
        verbose=verbose,
    )

    ## UNCOMMENT the following when https://github.com/neuronets/nobrainer/pull/125
    ## is merged
    # if not invalid_pairs:
    #     click.echo(click.style("Passed verification.", fg="green"))
    # else:
    #     click.echo(click.style("Failed verification.", fg="red"))
    #     for pair in invalid_pairs:
    #         click.echo(pair[0])
    #         click.echo(pair[1])
    #     sys.exit(-1)

    ppaths = preprocess_parallel(
        volume_filepaths,
        conform_volume_to=volume_shape,
        num_parallel_calls=num_parallel_calls,
        save_path=preprocess_path,
    )

    invalid_pairs = verify_features_labels(
        ppaths,
        volume_shape=volume_shape,
        check_labels_int=True,
        num_parallel_calls=num_parallel_calls,
        verbose=verbose,
    )

    if not invalid_pairs:
        click.echo(
        )
    else:
        click.echo(click.style("Failed post preprocessing re-verification.", fg="red"))
        click.echo(
            f"Oops! This is embarrasing. Looks like our preprocessing"
            " script shit the bed. Found {len(invalid_pairs)} invalid"
            " pairs of volumes. These files might not all have shape "
            " {volume_shape} or the labels might not be scalar values"
            " Please report this issue on                            "
            " https://github.com/poldracklab/nondefaced-detector     "
        )

        for pair in invalid_pairs:
            click.echo(pair[0])
            click.echo(pair[1])
        sys.exit(-1)

    # TODO: Convert to tfrecords
    os.makedirs(os.path.dirname(tfrecords_template), exist_ok=True)

    _write_tfrecord(
            features_labels=ppaths,
            filename_template=tfrecords_template,
            examples_per_shard=examples_per_shard,
            processes=num_parallel_calls,
            verbose=verbose,
    )

    click.echo(click.style("Finished conversion to TFRecords.", fg="green"))


@cli.command()
@click.argument("infile")
# @click.argument("outfile")
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to model weights. NOTE: A version of pretrained model weights can be found here: https://github.com/poldracklab/nondefaced-detector/tree/master/model_weights",
    **_option_kwds,
)
@click.option(
    "-t",
    "--classifier-threshold",
    default=0.5,
    type=float,
    help="Threshold for the classifier [Default is 0.5].",
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
    default="/tmp",
    help="Path to save preprocessed volumes.",
    **_option_kwds,
)
@click.option(
    "-j",
    "--num-parallel-calls",
    default=-1,
    type=int,
    help="Number of processes to use. If -1, uses all available processes.",
    **_option_kwds,
)
@click.option(
    "--skip-header", is_flag=True, help="Skip csv header.", **_option_kwds
)
@click.option(
    "--keep-preprocessed", is_flag=True, help="Keep the preprocessed volumes.", **_option_kwds
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress bar.", **_option_kwds
)
def predict(
    *,
    infile,
    classifier_threshold,
    model_path,
    conform_volume_to,
    preprocess_path,
    num_parallel_calls,
    skip_header,
    keep_preprocessed,
    verbose,
):
    """Predict labels from features using a trained model.

    The predictions are saved to OUTFILE.
    """

    if verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        import tensorflow as tf

        tf.get_logger().setLevel(logging.INFO)
        tf.autograph.set_verbosity(1)

    # if os.path.exists(outfile):
    #     raise FileExistsError(
    #         "Output file already exists. Will not overwrite {}".format(outfile)
    #     )

    if not os.path.exists(infile):
        raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filename
        )

    required_dirs = ["axial", "coronal", "sagittal", "combined"]

    for plane in required_dirs:
        if not os.path.isdir(os.path.join(model_path, plane)):
            raise ValueError("Missing {} directory in model path".format(plane))

    if infile.endswith('.nii') or infile.endswith('.nii.gz'):

        cpath = preprocess(
                infile,
                save_path=preprocess_path,
                with_label=False
        )

        volume, _, _ = utils.load_vol(cpath)
        model = prediction._get_model(model_path)
        predicted = prediction._predict(volume, model)

        print("Final layer output: ", predicted)
        print("Input classifier threshold: ", classifier_threshold)

        if predicted[0] >= classifier_threshold:
            print("Predicted Class: NONDEFACED")
        else:
            print("Predicted Class: DEFACED")

        if preprocess_path == "/tmp":
            cleanup_files(cpath)

    if infile.endswith('csv'):
        filepaths = []
        with open(infile, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            if skip_header:
                next(reader)

            for row in reader:
                filepaths.append(row[0])

        num_parallel_calls = None if num_parallel_calls == -1 else num_parallel_calls
        if num_parallel_calls is None:
            # Get number of processes allocated to the current process.
            # Note the difference from `os.cpu_count()`.
            num_parallel_calls = len(os.sched_getaffinity(0))

        outputs = preprocess_parallel(
                filepaths,
                num_parallel_calls=num_parallel_calls,
                conform_volume_to=conform_volume_to,
                with_label=False,
        )

        preds = prediction.predict(outputs, model_path=model_path, n_slices=32)

        with open('outputs.csv', 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['volume','score'])
            for row in preds:
                csv_out.writerow(row)

        if not keep_preprocessed:
            cleanup_files(*outputs)


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
