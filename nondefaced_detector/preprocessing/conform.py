"""Conforming input data."""

from pathlib import Path
from tempfile import mkdtemp
from scipy.ndimage import map_coordinates

import numpy as np
import nibabel as nb


def conform_data(
    in_file, out_file=None, out_size=(256, 256, 256), out_zooms=(1.0, 1.0, 1.0), order=3
):
    """Conform the input dataset to the canonical orientation.

    Parameters
    ----------
    in_file: str - Path
        Path to the input MRI volume to conform.
    out_file: str - Path, default=None
        Path to save the conformed volume. By default the
        volume is saved as /tmp/conformed.nii.gz
    out_size: tuple of size 3, optional, default=(256, 256, 256)
        The shape to conform the 3D volume to.
    out_zooms: tuple of size 3, optional, default=(1.0, 1.0, 1.0)
        Factors to normalize voxel size to.
    order: int, optional, default=3
        Order of the spline interpolation. The order has to be in
        the range 0-5.

    Returns
    -------
    str - Path
        The path to where the conformed volume is saved.
    """

    if isinstance(in_file, (str, Path)):
        in_file = nb.load(in_file)

    # Drop axes with just 1 sample (typically, a 3D file stored as 4D)
    in_file = nb.squeeze_image(in_file)
    dtype = in_file.header.get_data_dtype()

    # Reorient to closest canonical
    in_file = nb.as_closest_canonical(in_file)
    data = np.asanyarray(in_file.dataobj)

    # Calculate the factors to normalize voxel size to out_zooms
    normed = np.array(out_zooms) / np.array(in_file.header.get_zooms()[:3])

    # Calculate the new indexes, sampling at 1mm^3 with out_size sizes.
    # center_ijk = 0.5 * (np.array(in_file.shape) - 1)
    new_ijk = normed[:, np.newaxis] * np.array(
        np.meshgrid(
            np.arange(out_size[0]),
            np.arange(out_size[1]),
            np.arange(out_size[2]),
            indexing="ij",
        )
    ).reshape((3, -1))

    offset = 0.5 * (np.max(new_ijk, axis=1) - np.array(in_file.shape))

    # Align the centers of the two sampling extents
    new_ijk -= offset[:, np.newaxis]

    # Resample data in the new grid
    resampled = map_coordinates(
        data,
        new_ijk,
        output=dtype,
        order=order,
        mode="constant",
        cval=0,
        prefilter=True,
    ).reshape(out_size)

    resampled[resampled < 0] = 0

    # Create a new x-form affine, aligned with cardinal axes, 1mm3 and centered.
    newaffine = np.eye(4)
    newaffine[:3, 3] = -0.5 * (np.array(out_size) - 1)
    nii = nb.Nifti1Image(resampled, newaffine, None)
    if out_file is None:
        out_file = Path(mkdtemp()) / "conformed.nii.gz"
    out_file = Path(out_file).absolute()

    nii.to_filename(out_file)

    return out_file
