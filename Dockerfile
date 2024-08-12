FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Configure environment
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    TORCH_CUDA_ARCH_LIST="8.6"

# Redirect shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install prereqs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git-lfs \
    ffmpeg \ 
    libgl1-mesa-dev \
    libglib2.0-0 \
    git \
    python3-dev \
    python3-pip \
    # Lunar Tools prereqs
    libasound2-dev \
    libportaudio2 \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set symbolic links
RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash. bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-12.1" >> /etc/bash.bashrc


RUN apt-get update && apt-get install -y \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://github.com/lunarring/lunar_tools

RUN pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install https://github.com/chengzeyi/stable-fast/releases/download/v1.0.5/stable_fast-1.0.5+torch222cu121-cp310-cp310-manylinux2014_x86_64.whl

RUN pip3 install diffusers==0.26.3

# Clone base repo because why not
RUN git clone https://github.com/lunarring/lunar_tools.git
RUN git clone https://github.com/lunarring/rtd.git

# Install Python packages: Basic, then CUDA-compatible, then custom
RUN pip3 install \
    xformers>=0.0.22 \
    transformers==4.37.2 \
    accelerate==4.37.2 \
    triton>=2.1.0 

# Optionally store weights in image
# RUN git lfs install && git clone https://huggingface.co/stabilityai/sdxl-turbo /sdxl-turbo


