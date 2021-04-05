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


```bash
```

## Building from local Docker cache


```bash
```
