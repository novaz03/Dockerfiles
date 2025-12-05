#!/bin/bash

# 1. Start vLLM Server in the background
# We pass standard vLLM environment variables and arguments
echo "Starting vLLM server for model: $MODEL_NAME..."

# Build the command
# We bind to 0.0.0.0 to be safe, though localhost is used by the handler
CMD="python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code"

# Add extra arguments if VLLM_ARGS is set (e.g. --enable-reasoning --tensor-parallel-size 2)
if [ ! -z "$VLLM_ARGS" ]; then
    CMD="$CMD $VLLM_ARGS"
fi

# Run the command in the background
$CMD &
VLLM_PID=$!

# 2. Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
while ! curl -s http://localhost:8000/health > /dev/null; do
  sleep 2
  echo "Still waiting for vLLM..."
  
  # Check if process died
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died unexpectedly."
    exit 1
  fi
done

echo "vLLM is ready!"

# 3. Start Runpod Handler
# This function blocks and handles the serverless jobs
echo "Starting Runpod handler..."
python3 -u /app/handler.py

# 4. Cleanup (if handler exits)
kill $VLLM_PID
