# Pin to RunPod's CUDA/flash-attn tuned vLLM worker (matches runpod/worker-v1-vllm)
FROM runpod/worker-v1-vllm:v2.11.0

# flash-attn version is provided by the base image; override via pip only if you know
# a matching wheel exists for the bundled CUDA/torch combo.

USER root

# Install ffmpeg (for Whisper), curl (health checks), and git (flash-attn build fallback)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK, requests, faster-whisper, and a bleeding-edge Transformers (from GitHub)
# so qwen3/qwen3_moe architectures are recognized. Keep base vLLM/torch/flash-attn intact.
RUN pip install --no-cache-dir --upgrade --upgrade-strategy eager \
    runpod \
    requests \
    faster-whisper \
    "transformers @ git+https://github.com/huggingface/transformers.git" \
    tokenizers

# Environment variables
# MODEL_NAME can be overridden at runtime
ENV MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
ENV VLLM_ARGS=""
# ENV HF_TOKEN=""
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.95
ENV ENABLE_PREFIX_CACHING=true
ENV MAX_NUM_BATCHED_TOKENS=2048
ENV MAX_NUM_SEQS=256
ENV WHISPER_MODEL="large-v3-turbo"
ENV WHISPER_DEVICE="cuda"
ENV WHISPER_COMPUTE_TYPE="float16"
ENV PYTHONUNBUFFERED=1

# Work directory
WORKDIR /app

# Copy the handler, start script, and optional .env (if present) so start.sh can read it
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
COPY .env /app/.env

# Make the start script executable
RUN chmod +x /app/start.sh

# --- OPTIONAL: BAKE MODEL INTO IMAGE ---
# Uncomment the following lines if you want to download the model during build 
# (Reduces cold start time but makes image larger)
# ARG HF_TOKEN_BUILD
# ENV HF_TOKEN=$HF_TOKEN_BUILD
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct')"
# ---------------------------------------

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
