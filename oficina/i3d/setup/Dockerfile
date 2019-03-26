FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install opencv-python

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev git

WORKDIR /home/i3d/src

RUN git clone https://github.com/deepmind/kinetics-i3d /home/i3d/kinetics-i3d