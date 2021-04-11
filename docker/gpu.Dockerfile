FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get install software-properties-common -y && add-apt-repository ppa:git-core/ppa -y

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y vim wget

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version

RUN conda install -c conda-forge datalad -y

RUN git config --global user.email "detector@nondefaced.com"
RUN git config --global user.name "nondefaced-detector"

RUN datalad clone https://gin.g-node.org/shashankbansal56/nondefaced-detector-reproducibility /opt/nondefaced-detector-reproducibility

RUN cd /opt/nondefaced-detector-reproducibility

RUN datalad get pretrained_weights/*
RUN datalad get examples/*

ENV MODEL_PATH='/opt/nondefaced-detector-reproducibility/pretrained_weights'
ARG MODEL_PATH='/opt/nondefaced-detector-reproducibility/pretrained_weights'

RUN pip3 install --upgrade tensorflow-gpu==2.3.2
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
