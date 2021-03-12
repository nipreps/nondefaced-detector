FROM tensorflow/tensorflow:2.4.1-jupyter

RUN apt-get install -y vim

RUN pip3 install nobrainer \
                 sklearn \
                 pandas  \
                 seaborn \
                 numpy   \
                 matplotlib \
                 nibabel

COPY [".", "/opt/nondefaced-detector"]

RUN python3 -m pip install --no-cache-dir --editable /opt/nondefaced-detector

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ENV TF_CPP_MIN_LOG_LEVEL='3'

WORKDIR "/work"

LABEL maintainer="Shashank Bansal <shashankbansal56@gmail.com>"

ENTRYPOINT ["nondefaced-detector"]
