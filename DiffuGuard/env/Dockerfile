# GPU image for the DiffuGuard reproduction: CUDA 12.4 + Python 3.10 + DiffuGuard's pinned requirements.txt
# (byte-identical to DIJA's requirements.txt, so the DIJA venv / image works too).
# Build from the repo root (DLM_safety_code/, submodules must be checked out):
#   git submodule update --init
#   docker build -f DiffuGuard/env/Dockerfile -t dlm-safety/diffuguard-repro .
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

# av==14.4.0 is excluded: PyPI has no wheel for it (source build needs ffmpeg 7) and DiffuGuard never imports it.
COPY DiffuGuard/third_party/DiffuGuard/requirements.txt /tmp/dg-requirements.txt
RUN python -m pip install --upgrade pip \
    && grep -v '^av==' /tmp/dg-requirements.txt > /tmp/req.txt \
    && python -m pip install -r /tmp/req.txt \
    && python -m pip install PyYAML python-dotenv pytest

COPY . .

# Weights are NOT baked in. Mount them: -v <host>/hf_models:/workspace/DLM_safety_code/hf_models
CMD ["bash"]
