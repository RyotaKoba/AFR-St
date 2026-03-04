FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

LABEL Author="Ryota Kobayashi"
LABEL Version="1.0"
LABEL Description="ML environment with PyTorch, Transformers, and related libraries"

# 環境変数の設定
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:$PYTHONPATH
ARG USERNAME
ARG USERID

RUN useradd -ml -u ${USERID} -s /bin/bash -G sudo ${USERNAME} && \
    echo "${USERNAME}:password" | chpasswd

# システムの依存関係をインストール
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        sudo \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 13.0 support
RUN pip3 install --no-cache-dir --break-system-packages\
    torch==2.9.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Core ML and data processing libraries
RUN pip3 install --no-cache-dir --break-system-packages\
    transformers==4.57.1 \
    datasets==3.6.0 \
    tokenizers==0.22.2 \
    safetensors==0.6.2 \
    timm \
    accelerate==1.7.0

# Data manipulation libraries
RUN pip3 install --no-cache-dir --break-system-packages\
    numpy==1.26.4 \
    pandas \
    pyarrow

# Visualization libraries
RUN pip3 install --no-cache-dir --break-system-packages\
    matplotlib \
    pillow

# Utility libraries
RUN pip3 install --no-cache-dir --break-system-packages\
    huggingface-hub \
    requests \
    tqdm \
    pyyaml \
    psutil \
    regex \
    scikit-learn

# CLI tool
RUN pip3 install --no-cache-dir --break-system-packages clize

# デフォルトのコマンド
# ユーザーを切り替え
USER ${USERNAME}
CMD ["/bin/bash"]
