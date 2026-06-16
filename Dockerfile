# Serving image for the coin VLM (FastAPI + vLLM).
#
# Locks a CUDA 12 stack: vLLM < 0.20 ships a CUDA 12 wheel on PyPI as its
# default, while vLLM >= 0.20 defaults to a CUDA 13 build that needs a newer
# NVIDIA driver (libcudart.so.13). This image therefore runs on hosts with an
# NVIDIA driver >= 12.x (e.g. the A100 box on driver 12.4) via CUDA minor-version
# compatibility — no CUDA 13 driver required.
#
# Model weights and logs are NOT baked in; mount them at runtime (see below).
#
# Build:
#   docker build -t coin-vlm-serve .
#
# Run (GPU 3, model + logs mounted from host):
#   docker run --rm --gpus '"device=3"' -p 49710:49710 \
#     -v /data/coin/coin_vlm_finetune/data/models/Qwen3-VL-8B-coin-awq:/models/coin-awq:ro \
#     -v /data/coin/coin_vlm_finetune/outputs/logs/serving:/logs \
#     coin-vlm-serve
FROM docker.m.daocloud.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libglib2.0-0 libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) vLLM CUDA-12 build (PyPI default for <0.20) + matching torch (cu128).
#    Pinned <0.20 so we never pull the CUDA-13 default that needs a newer driver.
#    Needs PyPI + download.pytorch.org reachable during build (NOT github).
RUN python -m pip install --upgrade pip \
 && python -m pip install "vllm>=0.11,<0.20" \
        --extra-index-url https://download.pytorch.org/whl/cu128

# 2) Serving dependencies.
COPY requirements-serve.txt .
RUN python -m pip install -r requirements-serve.txt

# 3) Application code only (weights/outputs are mounted at runtime).
COPY src ./src
COPY scripts ./scripts
COPY config ./config

EXPOSE 49710

# model_path -> the mount point; logs -> mounted volume; ngrok off in containers.
CMD ["python", "scripts/serve.py", \
     "--data_config", "config/data/coin_dataset.yaml", \
     "--model_config", "config/model/qwen3_vl_8b.yaml", \
     "--training_config", "config/training/training.yaml", \
     "--serving_config", "config/serving/serving.yaml", \
     "--override", \
     "serving.model_path=/models/coin-awq", \
     "serving.log_dir=/logs", \
     "serving.ngrok.enabled=false", \
     "serving.host=0.0.0.0"]
