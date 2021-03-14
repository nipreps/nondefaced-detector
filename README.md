# Nondefaced-Detector
A framework to detect if a 3D MRI volume has been defaced.

## Table of contents
- [Installation](#installation)
  - [Container](#container)
    - [GPU](#gpu)
    - [CPU](#cpu)
  - [Pip](#pip)
- [Using pre-trained networks](#using-pre-trained-networks)
- [Reproducibility](#reproducibility)
- [Paper](#paper)
- [Roadmap](#roadmap)
- [Questions or Issues](#questions-or-issues)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  - [Training Dataset](#training-dataset)
  - [Built With](#built-with)

## Installation

### Container
We recommend using the latest *Nondefaced-detector* docker container, which includes all the dependencies required for the framework. 

**GPU**

The *Nondefaced-detector* GPU supported container uses the tensorflow-gpu as its base image. Please see the [official tensorflow docker install page](https://www.tensorflow.org/install/docker) for all of the CUDA and NVIDIA driver requirements.

```bash
$ docker pull poldracklab/nondefaced-detector:latest-gpu
```

**CPU**

This container can be used on most systems that have Docker/Singularity installed.

```bash
$ docker pull poldracklab/nondefaced-detector:latest-cpu
```
NOTE: The CPU container will be very slow for training. We highly recommend that you use a GPU system.

### Pip

```bash
$ pip install --no-cache-dir nondefaced-detector[gpu]
```

<!-- USAGE EXAMPLES -->
## Using pre-trained networks
Pre-trained networks are avalaible in the *Nondefaced-detector* [models](https://github.com/poldracklab/nondefaced-detector/tree/master/nondefaced_detector/models) repository. Prediction can be done using the nondefaced-detector CLI or in python.

### From docker container installation

```bash
$ docker run --rm -v $PWD:/data nondefaced-detector:latest-cpu \
    predict \
    --model-path=/opt/nondefaced-detector/nondefaced_detector/models/pretrained_weights \
    /data/example1.nii.gz
```

### From pip installation

```bash
$ nondefaced-detector
Usage: nondefaced-detector [OPTIONS] COMMAND [ARGS]...

  A framework to detect if a 3D MRI Volume has been defaced.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  convert   Preprocess MRI volumes and convert to Tfrecords.
  evaluate  Evaluate a model's predictions against known labels.
  info      Return information about this system.
  predict   Predict labels from features using a trained model.
```
<!-- USAGE EXAMPLES -->
## Reproducibility

Steps to reproduce inference results from the paper. 

**Step 1:** Get the preprocessed dataset. You need to have [datalad](https://handbook.datalad.org/en/latest/intro/installation.html) installed. 

```bash
$ datalad clone https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility /data/nondefaced-detector-reproducibility
$ cd /data/nondefaced-detector-reproducibility
$ datalad get test_ixi/tfrecords/*

```
NOTE: To reproduce inference results from the paper, you only need to download the tfrecords.

**Step 2:** Depending on your system create a tensorflow-cpu/gpu virtual environment. We recommend using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
$ conda create -n tf-cpu tensorflow 
$ conda activate tf-cpu
```

**Step 3:** Get the nondefaced-detector repo.

```bash
$ git clone https://github.com/poldracklab/nondefaced-detector.git
```
**Step 4:** Run the standalone inference script. The inference script uses the pre-trained model weights under `nondefaced_detector/models/pretrained_weights`
```bash
$ cd nondefaced-detector
$ pip install -e .
$ cd nondefaced_detector
$ python inference.py < PATH_TO_TFRECORDS [/data/nondefaced-detector-reproducibility/test_ixi/tfrecords] > 
```

## Paper

## Roadmap

See the [projects dashboard](https://github.com/poldracklab/nondefaced-detector/projects) for a list of ongoing work and proposed features. 

## Questions or Issues
See the [open issues](https://github.com/poldracklab/nondefaced-detector/issues) for a list of known issues. If you have any questions or encounter any issues, please submit a github issue. 


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Shashank Bansal - shashankbansal56@gmail.com 


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

### Training Dataset
The original model was trained on 980 defaced MRI scans from 36 different studies that are publicly available at [OpenNeuro.org](https://openneuro.org/)
### Built With

* [nobrainer](https://github.com/neuronets/nobrainer)
* [IXI Dataset](https://brain-development.org/ixi-dataset/)


<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/poldracklab/nondefaced-detector/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/poldracklab/nondefaced-detector/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/poldracklab/nondefaced-detector/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/poldracklab/nondefaced-detector/blob/master/LICENSE.txt
