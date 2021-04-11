"""Setup script for nondefaced-detector.
To install, run `python3 setup.py install`.
"""
from setuptools import setup

setup(
    name="nondefaced-detector",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
)
