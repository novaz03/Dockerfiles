#!/bin/bash
set -euo pipefail

APP_DIR="/app"

# Load .env so we can turn those settings into CLI flags
if [ -f "${APP_DIR}/.env" ]; then
  echo "Loading ${APP_DIR}/.env into environment..."
  set -a
  # shellcheck disable=SC1090
  source "${APP_DIR}/.env"
  set +a
fi

VLLM_ARGS_BUILT="${VLLM_ARGS:-}"

append_arg() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name-}"
  if [ -n "${value}" ]; then
    VLLM_ARGS_BUILT+=" ${flag} ${value}"
  fi
}

append_bool() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name-}"
  if [ -z "${value}" ]; then
    return
  fi
  case "${value}" in
    1|true|TRUE|yes|YES|on|ON)
      VLLM_ARGS_BUILT+=" ${flag}"
      ;;
    *)
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
  [GUIDED_DECODING_BACKEND]="--guided-decoding-backend"
  [PIPELINE_PARALLEL_SIZE]="--pipeline-parallel-size"
  [TENSOR_PARALLEL_SIZE]="--tensor-parallel-size"
  [MAX_PARALLEL_LOADING_WORKERS]="--max-parallel-loading-workers"
  [ENABLE_PREFIX_CACHING]="--enable-prefix-caching"
  [GPU_MEMORY_UTILIZATION]="--gpu-memory-utilization"
  [BLOCK_SIZE]="--block-size"
  [SWAP_SPACE]="--swap-space"
  [MAX_SEQ_LEN_TO_CAPTURE]="--max-seq-len-to-capture"
  [MAX_NUM_BATCHED_TOKENS]="--max-num-batched-tokens"
  [MAX_NUM_SEQS]="--max-num-seqs"
  [MAX_LOGPROBS]="--max-logprobs"
  [QUANTIZATION]="--quantization"
  [SEED]="--seed"
  [ENFORCE_EAGER]="--enforce-eager"
)

BOOLEAN_VARS=(
  SKIP_TOKENIZER_INIT
  TRUST_REMOTE_CODE
  ENABLE_PREFIX_CACHING
  ENFORCE_EAGER
)

for var in "${!FLAG_MAP[@]}"; do
  if printf '%s\n' "${BOOLEAN_VARS[@]}" | grep -qx "${var}"; then
    append_bool "${var}" "${FLAG_MAP[$var]}"
  else
    append_arg "${var}" "${FLAG_MAP[$var]}"
  fi
done

# Trim leading whitespace
VLLM_ARGS_BUILT="${VLLM_ARGS_BUILT#" "}"

echo "Starting vLLM server for model: ${MODEL_NAME}..."
echo "vLLM args: ${VLLM_ARGS_BUILT}"

# Build the command; bind to 0.0.0.0 though handler uses localhost
CMD="python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port 8000 \
    --host 0.0.0.0"

if [ -n "${VLLM_ARGS_BUILT}" ]; then
  CMD="${CMD} ${VLLM_ARGS_BUILT}"
fi

# Run the command in the background
eval "${CMD}" &
VLLM_PID=$!

# 2. Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
until curl -s http://localhost:8000/health > /dev/null; do
  sleep 2
  echo "Still waiting for vLLM..."

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM process died unexpectedly."
    exit 1
  fi
done

echo "vLLM is ready!"

# 3. Start Runpod Handler
echo "Starting Runpod handler..."
python3 -u /app/handler.py

# 4. Cleanup (if handler exits)
kill "${VLLM_PID}"
