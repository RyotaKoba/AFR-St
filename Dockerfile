FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

LABEL Author="Ryota Kobayashi"
LABEL Version="1.0"
LABEL Description="ML environment with PyTorch, Transformers, and related libraries"

# 環境変数の設定
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:$PYTHONPATH

RUN useradd -ml -u 1001 -s /bin/bash -G sudo ryotakoba && \
    echo "ryotakoba:password" | chpasswd



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
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Core ML and data processing libraries
RUN pip3 install --no-cache-dir --break-system-packages\
    transformers \
    datasets \
    tokenizers \
    safetensors \
    timm \
    accelerate

# Data manipulation libraries
RUN pip3 install --no-cache-dir --break-system-packages\
    numpy \
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
    regex

# CLI tool
RUN pip3 install --no-cache-dir --break-system-packages clize

# デフォルトのコマンド
# ユーザーを切り替え
USER ryotakoba
CMD ["/bin/bash"]
