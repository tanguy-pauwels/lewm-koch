FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HYDRA_FULL_ERROR=1 \
    STABLEWM_HOME=/workspace/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /workspace/requirements.txt

COPY . /workspace
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
RUN mkdir -p /workspace/data

ENTRYPOINT ["python3", "/workspace/train_wrapper.py"]
