#!/usr/bin/env bash
# ======================================================================
# vLLM 部署 + TTFT / ITL（与首次 TP=2 口径相同，只改 TP / GPU）
# ======================================================================
# 首次已测：
#   GPUS=0,1 TP_SIZE=2  → /tmp/vllm_tp2_ttft_itl.json
# 对齐对照：
#   GPUS=4 TP_SIZE=1    → 其余默认不变
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/Qwen3.5-9B}"
GPUS="${GPUS:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-18101}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-8}"
CONCURRENCY="${CONCURRENCY:-1}"
WAIT_READY_S="${WAIT_READY_S:-600}"
OUT_JSON="${OUT_JSON:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[bench] ERROR: 找不到 python3 / python" >&2
  exit 2
fi

IFS=',' read -r -a _GPU_ARR <<< "${GPUS}"
if [[ "${#_GPU_ARR[@]}" -lt "${TP_SIZE}" ]]; then
  echo "[bench] ERROR: GPUS=${GPUS} 少于 TP_SIZE=${TP_SIZE}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
URL="http://127.0.0.1:${PORT}/v1"

echo "[bench] PYTHON_BIN=${PYTHON_BIN}"
echo "[bench] MODEL=${MODEL_PATH}"
echo "[bench] GPUS=${GPUS} TP=${TP_SIZE} PORT=${PORT} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[bench] prompt_tokens=${PROMPT_TOKENS} max_tokens=${MAX_TOKENS} warmup=${WARMUP} runs=${RUNS} conc=${CONCURRENCY}"

LOG="/tmp/vllm_tp2_bench_${PORT}.log"
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --dtype "${DTYPE}" \
  >"${LOG}" 2>&1 &
SRV_PID=$!
echo "[bench] server pid=${SRV_PID} log=${LOG}"

cleanup() {
  if kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "[bench] stopping server pid=${SRV_PID}"
    kill "${SRV_PID}" 2>/dev/null || true
    wait "${SRV_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! kill -0 "${SRV_PID}" 2>/dev/null; then
  echo "[bench] ERROR: server exited immediately; tail ${LOG}" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 1
fi

PY_ARGS=(
  --url "${URL}"
  --model "${MODEL_PATH}"
  --prompt-tokens "${PROMPT_TOKENS}"
  --max-tokens "${MAX_TOKENS}"
  --warmup "${WARMUP}"
  --runs "${RUNS}"
  --concurrency "${CONCURRENCY}"
  --wait-ready-s "${WAIT_READY_S}"
)
if [[ -n "${OUT_JSON}" ]]; then
  PY_ARGS+=(--out-json "${OUT_JSON}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/bench_vllm_ttft_itl.py" "${PY_ARGS[@]}"
