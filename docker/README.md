# Nondefaced-detector in a container

The Dockerfiles in this directory can be used to create Docker images to use _Nondefaced-detector_ on CPU or GPU.

## Build images

```bash
cd /code/nondefaced-detector  # Top-level nondefaced-detector directory
python setup.py --version  # Ensure that a version submodule is generated (setuptools-scm)
docker build -t nondefaced-detector:master-cpu -f docker/cpu.Dockerfile .
docker build -t nondefaced-detector:master-gpu -f docker/gpu.Dockerfile .
```

# Convert Docker images to Singularity containers

Using Singularity version 3.x, Docker images can be converted to Singularity containers using the `singularity` command-line tool.

## Pulling from DockerHub

In most cases (e.g., working on a HPC cluster), the nondefaced-detector singularity container can be created with:

```bash
sudo singularity pull docker://shashankbansal56/nondefaced-detector:latest-gpu
sudo singularity pull docker://shashankbansal56/nondefaced-detector:latest-cpu

```

## Building from local Docker cache

If you built a nondefaced-detector docker image locally and would like to convert it to a Singularity container, you can do so with:

```bash
sudo docker save <image_id> -o local.tar
sudo singularity build nondefaced-detector.sif docker-archive://local.tar
```

## Pre-built singularity images

You can also find pre-built singularity images here: [https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility/src/master/singularity](https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility/src/master/singularity)
