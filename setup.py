import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='MRIfaceDetector',
    version='0.0.1',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/poldracklab/mri-face-detector",
    packages=setuptools.find_packages(),
    install_requires=[
         'nibabel',
         'SimpleITK',
         'tensorflow',
         'numpy',
         'imgaug',
         'scikit-learn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
