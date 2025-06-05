# ------------------------------------------------------------------------------
# 以 NVIDIA 官方提供的 CUDA 11.8 + cuDNN8 Ubuntu 22.04 為基底
# ------------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# （可選）設定時區為 Asia/Taipei
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# ------------------------------------------------------------------------------
# 1. 安裝 Python 3.10、pip 及各項系統套件（對應原本 setup.sh 之指令）
# ------------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        python3-pip \
        libgl1 \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 2. 建立工作目錄（後面可自行掛載本機程式碼到 /workspace）
# ------------------------------------------------------------------------------
WORKDIR /workspace

# ------------------------------------------------------------------------------
# 3. 直接在這裡把 requirements.txt 的內容寫死：
#    （先安裝 CPU 版 paddlepaddle，其他 Python 套件一次到位）
# ------------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    opencv-python \
    pillow \
    paddlepaddle==2.5.2 \
    paddleocr==2.10.0 \
    matplotlib \
    reportlab \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    layoutparser

# ------------------------------------------------------------------------------
# 4. 安裝 PyTorch + CUDA 11.8（cu118）對應版本
# ------------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    torch==2.3.1+cu118 \
    torchvision==0.18.1+cu118 \
    torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------------------------
# 5. 安裝 paddlepaddle-gpu（注意：先有 CPU 版 paddlepaddle，再以 GPU 版覆蓋）
# ------------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
    paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# ------------------------------------------------------------------------------
# 6. 暴露容器預設要使用的埠（可依需求自行在 docker run 時調整，不寫也沒關係）
# ------------------------------------------------------------------------------
EXPOSE 8002

# ------------------------------------------------------------------------------
# 7. 容器啟動後的預設工作路徑
# ------------------------------------------------------------------------------
WORKDIR /workspace

# ------------------------------------------------------------------------------
# 8. 容器啟動後進入 bash，方便使用者互動
# ------------------------------------------------------------------------------
CMD ["bash"]
