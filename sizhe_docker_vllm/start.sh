#!/bin/bash
set -euo pipefail

APP_DIR="/app"
VOLUME_PATH="${BASE_PATH:-/runpod-volume}"
MODELS_CACHE="${MODELS_CACHE_DIR:-${VOLUME_PATH}/models}"

# --- 1. Cleanup / Signal Trap ---
cleanup() {
  echo "Shutting down..."
  if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM..."
    kill -TERM "${VLLM_PID}"
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  exit "${1:-0}"
}
trap 'cleanup 1' SIGTERM SIGINT

# --- 2. Environment Setup ---

# --- 3. Model Path Detection ---
MODEL_PATH="${MODEL_NAME}"
MODEL_SHORT_NAME=$(basename "${MODEL_NAME}")

echo "Looking for cached model..."
if [ -d "${MODELS_CACHE}/${MODEL_NAME}" ]; then
  MODEL_PATH="${MODELS_CACHE}/${MODEL_NAME}"
  echo "✓ Found cached model at: ${MODEL_PATH}"
elif [ -d "${MODELS_CACHE}/${MODEL_SHORT_NAME}" ]; then
  MODEL_PATH="${MODELS_CACHE}/${MODEL_SHORT_NAME}"
  echo "✓ Found cached model at: ${MODEL_PATH}"
elif [ -d "/app/models/${MODEL_SHORT_NAME}" ]; then
  MODEL_PATH="/app/models/${MODEL_SHORT_NAME}"
  echo "✓ Found baked-in model at: ${MODEL_PATH}"
else
  echo "⚠ Model not found in cache. Will download from HuggingFace: ${MODEL_NAME}"
fi

# --- 4. Argument Parsing (Using Arrays) ---
# Initialize as an empty array
VLLM_CMD_ARGS=()

append_arg() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name-}"
  # Check for empty or "None" values
  if [ -n "${value}" ]; then
    case "${value}" in
      None|none|null|NULL|"") return ;;
    esac
    # Append as distinct array elements (Preserves spaces!)
    VLLM_CMD_ARGS+=("${flag}" "${value}")
  fi
}

append_bool() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name-}"
  if [ -z "${value}" ]; then return; fi
  case "${value}" in
    1|true|TRUE|yes|YES|on|ON)
      VLLM_CMD_ARGS+=("${flag}")
      ;;
  esac
}

declare -A FLAG_MAP=(
  [TOKENIZER_MODE]="--tokenizer-mode"
  [SKIP_TOKENIZER_INIT]="--skip-tokenizer-init"
  [TRUST_REMOTE_CODE]="--trust-remote-code"
  [LOAD_FORMAT]="--load-format"
  [DTYPE]="--dtype"
  [KV_CACHE_DTYPE]="--kv-cache-dtype"
  [MAX_MODEL_LEN]="--max-model-len"
  [PIPELINE_PARALLEL_SIZE]="--pipeline-parallel-size"
  [TENSOR_PARALLEL_SIZE]="--tensor-parallel-size"
  [MAX_PARALLEL_LOADING_WORKERS]="--max-parallel-loading-workers"
  [ENABLE_PREFIX_CACHING]="--enable-prefix-caching"
  [GPU_MEMORY_UTILIZATION]="--gpu-memory-utilization"
  [BLOCK_SIZE]="--block-size"
  [SWAP_SPACE]="--swap-space"
  [MAX_NUM_BATCHED_TOKENS]="--max-num-batched-tokens"
  [MAX_NUM_SEQS]="--max-num-seqs"
  [MAX_LOGPROBS]="--max-logprobs"
  [QUANTIZATION]="--quantization"
  [SEED]="--seed"
  [ENFORCE_EAGER]="--enforce-eager"
  [SERVED_MODEL_NAME]="--served-model-name"
  [CHAT_TEMPLATE]="--chat-template"
)

# Fast boolean lookup
declare -A BOOLEAN_FLAGS=(
  [SKIP_TOKENIZER_INIT]=1
  [TRUST_REMOTE_CODE]=1
  [ENABLE_PREFIX_CACHING]=1
  [ENFORCE_EAGER]=1
)

for var in "${!FLAG_MAP[@]}"; do
  if [[ -n "${BOOLEAN_FLAGS[$var]-}" ]]; then
    append_bool "${var}" "${FLAG_MAP[$var]}"
  else
    append_arg "${var}" "${FLAG_MAP[$var]}"
  fi
done

echo "=========================================="
echo "Starting vLLM server"
echo "  Model: ${MODEL_PATH}"
# echo args safely (handle empty array)
if [ ${#VLLM_CMD_ARGS[@]} -gt 0 ]; then
  echo "  Args: ${VLLM_CMD_ARGS[*]}"
else
  echo "  Args: (none)"
fi
echo "=========================================="

# --- 5. Execution (No eval) ---
# Pass the array elements directly to the command
# Use ${VLLM_CMD_ARGS[@]+"${VLLM_CMD_ARGS[@]}"} to handle empty array safely
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port 8000 \
    --host 0.0.0.0 \
    ${VLLM_CMD_ARGS[@]+"${VLLM_CMD_ARGS[@]}"} &

VLLM_PID=$!

# --- 6. Health Check ---
echo "Waiting for vLLM to start..."
MAX_WAIT=600
WAITED=0

until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "❌ vLLM process died unexpectedly."
    wait "${VLLM_PID}" # Print exit code/stderr
    exit 1
  fi
  
  if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
    echo "❌ Timeout waiting for vLLM after ${MAX_WAIT} seconds."
    cleanup 1
  fi
  
  sleep 2
  WAITED=$((WAITED + 2))
  [ $((WAITED % 30)) -eq 0 ] && echo "Still waiting for vLLM... (${WAITED}s elapsed)"
done

echo "✓ vLLM is ready and healthy!"
# ---- Make pip-installed cuDNN discoverable for GPU Whisper ----
CUDNN_PIP_LIB=$(python3 - <<'PY'
import os
try:
    import nvidia.cudnn
    base = os.path.dirname(nvidia.cudnn.__file__)
    lib = os.path.join(base, "lib")
    print(lib if os.path.isdir(lib) else "")
except Exception:
    print("")
PY
)

if [ -n "${CUDNN_PIP_LIB}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PIP_LIB}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  echo "Using cuDNN from: ${CUDNN_PIP_LIB}"
else
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  echo "WARNING: pip cuDNN not found"
fi

python3 - <<'PY'
import ctypes, sys
names = ["libcudnn_ops.so.9.1.0","libcudnn_ops.so.9.1","libcudnn_ops.so.9"]
for n in names:
    try:
        ctypes.CDLL(n)
        print("Loaded:", n)
        sys.exit(0)
    except OSError:
        pass
print("Could not load cuDNN 9 ops library.")
sys.exit(1)
PY


# --- 7. Start Handler ---
echo "Starting Runpod handler..."
python3 -u /app/handler.py
HANDLER_EXIT=$?

echo "Handler exited with code ${HANDLER_EXIT}"
cleanup "${HANDLER_EXIT}"