# Pin to RunPod's CUDA/flash-attn tuned vLLM worker (matches runpod/worker-v1-vllm)
FROM runpod/worker-v1-vllm:stable-cuda12.1.0

ARG FLASH_ATTN_VERSION=2.6.1
ENV FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION}

USER root

# Install ffmpeg (for Whisper) and curl (for health checks)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK, requests, faster-whisper, and pin flash-attn to the base CUDA toolchain
RUN pip install --no-cache-dir runpod requests faster-whisper "flash-attn==${FLASH_ATTN_VERSION}"

# Environment variables
# MODEL_NAME can be overridden at runtime
ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
ENV VLLM_ARGS=""
# ENV HF_TOKEN=""
ENV WHISPER_MODEL="large-v3-turbo"
ENV WHISPER_DEVICE="cuda"
ENV WHISPER_COMPUTE_TYPE="float16"
ENV PYTHONUNBUFFERED=1

# Work directory
WORKDIR /app

# Copy the handler and start script
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh

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
