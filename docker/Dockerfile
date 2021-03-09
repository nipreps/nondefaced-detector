FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get install -y vim

RUN pip3 install --upgrade tensorflow-gpu==2.3.2
RUN pip3 install nobrainer \
                 sklearn \
                 pandas  \
                 seaborn \
                 numpy   \
                 matplotlib \
                 nibabel
                
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

