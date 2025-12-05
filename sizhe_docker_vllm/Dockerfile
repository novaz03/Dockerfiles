# Use the official vLLM image to ensure latest model support (Qwen3)
# Check https://hub.docker.com/r/vllm/vllm-openai/tags for specific versions
FROM vllm/vllm-openai:latest

# Install RunPod SDK and requests for the handler
RUN pip install --no-cache-dir runpod requests

# Environment variables
# MODEL_NAME can be overridden at runtime
ENV MODEL_NAME="Qwen/Qwen3-8B-Instruct"
ENV VLLM_ARGS=""
ENV HF_TOKEN=""

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
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-8B-Instruct')"
# ---------------------------------------

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
