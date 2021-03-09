<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/poldracklab/nondefaced-detector">
  </a>

  <h3 align="center">Nondefaced-Detector</h3>

  <p align="center">
    A framework to detect if a 3D MRI Volume has been defaced. 
    <br />
    <a href="https://github.com/poldracklab/nondefaced-detector"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/poldracklab/nondefaced-detector/issues">Report Bug</a>
    ·
    <a href="https://github.com/poldracklab/nondefaced-detector/issues">Request Feature</a>
  </p>
</p>


## Table of contents

- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)


<!-- ABOUT THE PROJECT -->
## About The Project

### Built With

* [nobrainer](https://github.com/neuronets/nobrainer)


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

### Installation


<!-- USAGE EXAMPLES -->
## Usage

<!-- USAGE EXAMPLES -->
## Reproducibility

Steps to reproduce inference results from the paper. 

**Step 1:** Get the preprocessed dataset. You need to have [datalad](https://handbook.datalad.org/en/latest/intro/installation.html) installed. 

```bash
$ datalad clone https://gin.g-node.org/shashankbansal56/nondefaced-detector-repoducibility.git
$ cd nondefaced-detector-reproducibility
$ datalad get text_ixi/tfrecords/*

```
NOTE: To reproduce inference results from the paper, you only need to download the tfrecords.

**Step 2:** Clone the nondefaced-detector repository.

```bash
$ git clone https://github.com/poldracklab/nondefaced-detector.git

```
**Step 3:** Run the standalone inference script. The inference script uses the pre-trained model weights under `nondefaced_detector/models/pretrained_weights`
```bash
$ cd nondefaced-detector/nondefaced_detector
$ python inference.py <PATH_TO_TFRECORDS>
```

<!-- ROADMAP -->
## Roadmap

See the [projects dashboard](https://github.com/poldracklab/nondefaced-detector/projects) for a list of ongoing work and proposed features. 
See the [open issues](https://github.com/poldracklab/nondefaced-detector/issues) for a list of known issues.



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
