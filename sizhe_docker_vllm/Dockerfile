FROM runpod/worker-v1-vllm:v2.11.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

# Keep base vLLM/torch/flash-attn intact; avoid eager upgrades
RUN pip install --no-cache-dir \
    runpod \
    requests \
    faster-whisper

# Pin to a known-good Transformers/tokenizers combo for Qwen + vLLM 0.11.x
RUN pip install --no-cache-dir \
    "transformers==4.57.1" \
    "tokenizers>=0.19.1" \
    flashinfer


ENV MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
ENV VLLM_ARGS=""
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.95
ENV ENABLE_PREFIX_CACHING=true
ENV MAX_NUM_BATCHED_TOKENS=2048
ENV MAX_NUM_SEQS=256
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
# COPY .env /app/.env  # consider not baking secrets
RUN chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
