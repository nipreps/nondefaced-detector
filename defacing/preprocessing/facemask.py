"""Utilities around binary face-masks."""
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
import nibabel as nb


def augment(in_file, facemask):
    """
    Generate augmentations from (MRI, face mask) pairs, where the face mask is mis-oriented.

    Parameters
    ----------
    in_file : :obj:`str` or path-like
        The anatomical MRI preserving facial features (i.e., unmasked).
    facemask : :obj:`str` or path-like
        The face-mask estimated elsewhere (e.g., with pydeface). Mask is 1 where facial
        features are present and 0 elsewhere.

    """
    retval = []  # A list where we accumulate the paths of augmentations
    retdir = Path(mkdtemp())
    # Some preliminary work: ensure we are getting NIfTI objects
    if isinstance(in_file, (str, Path)):
        in_file = nb.load(in_file)
    if isinstance(facemask, (str, Path)):
        facemask = nb.load(facemask)

    data = np.asanyarray(in_file.dataobj)
    msk = np.asanyarray(facemask.dataobj) > 0
    mskhdr = facemask.header.copy()
    mskhdr.set_data_dtype("uint8")

    # Unless the original facemask is incorrect, the AP axis should be the shortest
    bbox = _boundingbox(msk)
    # Reassign the index for the anterior-posterior (AP) and inferior-superior (SI) axes
    # (i.e., drop left-right). AP will be 1 and SI will be 2 when image canonically oriented.
    ap, _, si = [ax for _, ax in sorted(zip(bbox.shape, list(range(3))))]

    for i, flip_ap in enumerate((False, True)):
        newmask = msk.astype("uint8").copy()
        if flip_ap:
            newmask = np.flip(newmask, axis=ap)
        for j, flip_si in enumerate((False, True)):
            if flip_ap:
                newmask = np.flip(newmask, axis=si)

            # Apply mask
            defaced = data.copy()
            defaced[newmask == 1] = 0

            out_file = retdir / f"defaced{i:02}{j:02}.nii.gz"
            out_mask = retdir / f"facemsk{i:02}{j:02}.nii.gz"
            retval.append((out_file, out_mask))
            nb.Nifti1Image(newmask, facemask.affine, mskhdr).to_filename(out_mask)
            nb.Nifti1Image(defaced, in_file.affine, in_file.header).to_filename(out_file)
    return retval


def _boundingbox(mask):
    """Calculate the bounding box of a given binary mask."""
    bbox = np.argwhere(mask)
    return mask[
        bbox.min(0):bbox.max(0),
        bbox.min(1):bbox.max(1),
        bbox.min(2):bbox.max(2)
    ]
