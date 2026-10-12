# GPU image for the DIJA reproduction: CUDA 12.4 + Python 3.10 + DIJA's pinned requirements.txt
# Build from the repo root (submodule must be checked out):
#   git submodule update --init
#   docker build -f env/Dockerfile -t dlm-safety/dija-repro .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/DLM_safety_code

# Original requirements (torch==2.6.0 -> cu124 wheels on Linux)
COPY third_party/DIJA/requirements.txt /tmp/dija-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/dija-requirements.txt \
    && python -m pip install PyYAML python-dotenv pytest

COPY . .

# Weights are NOT baked in. Mount them: -v /data/hf_models:/workspace/DLM_safety_code/hf_models
CMD ["bash"]
