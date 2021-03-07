"""Main command-line interface for nondefacedDetector."""

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
from nondefaced_detector import __version__

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

    """ 
    1. Preprocess MRI volumes
    2. Convert preprocessed volumes to TFRecords.
    NOTE: Volumes must all be the same shape.
    """
    volume_filepaths = _read_csv(csv)
    return
if __name__ == "__main__":
    cli()
