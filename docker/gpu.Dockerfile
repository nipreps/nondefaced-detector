FROM ubuntu:20.04

RUN apt-get update -y && apt-get upgrade -y

RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y vim \
			wget \
			netbase \
			python3 \
			python3-pip \
			git \
			git-annex

COPY [".", "/opt/nondefaced-detector"]

RUN pip3 install --no-cache-dir --editable "/opt/nondefaced-detector[gpu]"

RUN git config --global user.email "detector@nondefaced.com"
RUN git config --global user.name "nondefaced-detector"

RUN datalad clone https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility /opt/nondefaced-detector-reproducibility

RUN datalad get /opt/nondefaced-detector-reproducibility/pretrained_weights/* -d /opt/nondefaced-detector-reproducibility
RUN datalad get /opt/nondefaced-detector-reproducibility/examples/* -d /opt/nondefaced-detector-reproducibility

ENV MODEL_PATH="/opt/nondefaced-detector-reproducibility/pretrained_weights"
ARG MODEL_PATH="/opt/nondefaced-detector-reproducibility/pretrained_weights"

ENV EXAMPLE_PATH="/opt/nondefaced-detector-reproducibility/examples"
ARG EXAMPLE_PATH="/opt/nondefaced-detector-reproducibility/examples"

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ENV TF_CPP_MIN_LOG_LEVEL='3'

WORKDIR "/work"

LABEL maintainer="Shashank Bansal <shashankbansal56@gmail.com>"

ENTRYPOINT ["nondefaced-detector"]
