FROM runpod/worker-v1-vllm:v2.11.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

# Core deps
RUN python3 -m pip install --no-cache-dir \
    runpod \
    requests \
    faster-whisper

# Install cuDNN 9 for CUDA 12 (GPU Whisper)
RUN python3 -m pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    "nvidia-cudnn-cu12>=9.1,<9.2"

# Pin HF stack for Qwen + vLLM 0.11.x
RUN python3 -m pip install --no-cache-dir \
    "transformers==4.57.1" \
    "tokenizers>=0.19.1"

# Disable FlashInfer entirely
RUN python3 -m pip uninstall -y flashinfer flashinfer-python || true

# Build-time proof cuDNN pkg exists
# RUN python3 -c "import nvidia.cudnn as c; print('cuDNN pkg:', c.__file__)"

ENV MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
ENV VLLM_ARGS=""
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.90
ENV ENABLE_PREFIX_CACHING=true
ENV MAX_NUM_BATCHED_TOKENS=2048
ENV MAX_NUM_SEQS=64

ENV WHISPER_MODEL="large-v3-turbo"
ENV WHISPER_DEVICE="cuda"
ENV WHISPER_COMPUTE_TYPE="float16"

ENV PYTHONUNBUFFERED=1

ENV BASE_PATH=/runpod-volume
ENV MODELS_CACHE_DIR=/runpod-volume/models
RUN mkdir -p /runpod-volume /runpod-volume/models

WORKDIR /app
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
