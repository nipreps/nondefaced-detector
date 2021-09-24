FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

COPY [".", "/opt/nondefaced-detector"]

RUN pip3 install --no-cache-dir --editable "/opt/nondefaced-detector"

RUN pip3 install pandas plotly matplotlib nibabel nobrainer numpy

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ENV TF_CPP_MIN_LOG_LEVEL='3'

LABEL maintainer="Shashank Bansal <shashankbansal56@gmail.com>"
