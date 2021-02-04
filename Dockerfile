FROM tensorflow/tensorflow:nightly-gpu-jupyter
RUN python3 -m pip install --no-cache-dir /opt/nobrainer
RUN pip3 install sklearn pandas
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR /app
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
