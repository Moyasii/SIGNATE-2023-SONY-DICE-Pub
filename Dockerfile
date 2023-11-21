FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    jupyter==1.0.0 \
    opencv_python==4.8.1.78 \
    matplotlib==3.7.3 \
    seaborn==0.13.0 \
    scikit-learn==1.3.1 \
    timm==0.9.8 \
    albumentations==1.3.1 \
    omegaconf==2.3.0 \
    torchmetrics==0.11.4 \
    lightning==2.1.0

RUN apt-get autoremove -y &&\
    apt-get clean

WORKDIR /work
COPY . /work