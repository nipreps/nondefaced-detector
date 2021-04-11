"""Utilities for Nondefaced-detector."""


import os
import tempfile

import datalad.api


_cache_dir = os.path.join(tempfile.gettempdir(), "nondefaced-detector-reproducibility")


def get_datalad(
    cache_dir=_cache_dir,
    datalad_repo="https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility",
    examples=False,
    test_ixi=False,
):
    """Download a datalad dataset/repo.

    The weights can be found at
    https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility/

    Parameters
    ----------
    cache_dir: str, directory where to clone datalad repo. Save to a /tmp by default

    """

    os.makedirs(cache_dir, exist_ok=True)

    try:
        datalad.api.clone(path=cache_dir, source=datalad_repo)
        datalad.api.get(
            path=os.path.join(cache_dir, "pretrained_weights"),
            dataset=cache_dir,
            recursive=True,
        )

        if examples:
            datalad.api.get(
                path=os.path.join(cache_dir, "examples"),
                dataset=cache_dir,
                recursive=True,
            )

        if test_ixi:
            inp = str(
                input(
                    "The test_ixi subdirectory contains large files and will take a while to download. \
                    Are you sure you want to download these? [y/n]"
                )
            )

            if "y" in inp.lower():
                datalad.api.get(
                    path=os.path.join(cache_dir, "test_ixi"),
                    dataset=cache_dir,
                    recursive=True,
                )

        return cache_dir

    except Exception as e:
        print(e)
        print("Something went wrong! Cleaning up...")
