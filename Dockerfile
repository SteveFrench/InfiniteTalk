# PyTorch/CUDA base suitable for Torch 2.4.x + cu121 wheels
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Workdir and code
WORKDIR /workspace
COPY . /workspace

# Python deps (pin to repo-compatible versions)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 && \
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        runpod requests librosa misaki[en] ninja psutil packaging wheel \
        flash_attn==2.7.4.post1 huggingface_hub hf_transfer && \
    pip install --no-cache-dir -r requirements.txt

# Speed up HF downloads (optional; harmless if hf_transfer isn't used)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# (Optional) create common dirs; in Serverless your volume mounts at /runpod-volume
RUN mkdir -p /runpod-volume/weights /workspace/outputs

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "-c", "import runpod, rp_handler; runpod.serverless.start({'handler': rp_handler.handler})"]
