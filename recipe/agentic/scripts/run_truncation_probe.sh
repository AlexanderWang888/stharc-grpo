#!/usr/bin/env bash
# ======================================================================
# 截断探针：少量 chain（默认 1 step），只看 truncation
# ======================================================================
#   bash recipe/agentic/scripts/run_truncation_probe.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_DIR}"

_detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

# 强制探针设定，避免 shell 残留盖掉（ALLOW_ENV_OVERRIDE=1 可放行）
if [[ "${ALLOW_ENV_OVERRIDE:-0}" != "1" ]]; then
  export MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B
  export DATA_DIR=data/stharc_public_smoke
  _NG="$(_detect_gpus)"
  if [[ -z "${_NG}" || "${_NG}" -lt 1 ]]; then
    _NG=4
  fi
  export NUM_GPUS="${_NG}"
  export TP_SIZE=1
  export GPU_MEM_UTIL=0.45
  # BATCH×N 必须能被 NUM_GPUS 整除；4 卡 → 2×2=4 条 chain / step
  export ROLLOUT_N=2
  export BATCH_SIZE=$(( NUM_GPUS / ROLLOUT_N ))
  if (( BATCH_SIZE < 1 )); then
    export BATCH_SIZE=1
    export ROLLOUT_N="${NUM_GPUS}"
  fi
  export NUM_AGENT_WORKERS=$((BATCH_SIZE * ROLLOUT_N))
  export VAL_BATCH_SIZE="${NUM_AGENT_WORKERS}"
  export TOTAL_STEPS=2   # 4 卡时约 8 条；贴近「跑五条」
  export SAVE_FREQ=0
  export TEST_FREQ=0
  # 预算按 parquet 真实 max K（不人为 K_MAX_CAP=12）；超长只打点不硬拦
  export K_MAX_CAP=999
  export STRICT_CHAIN_BUDGET=0
  export MAX_PROMPT_LEN=4096
  export MAX_RESP_LEN=2048
  export MAX_TASK_CONTEXT_LEN=4096
  unset MAX_MODEL_LEN CHAIN_RESP_LEN PPO_MAX_TOKEN_LEN K_MAX || true
  export MAX_NUM_SEQS=16
  export STHARC_RUN_FROZEN=false
  export ABORT_ON_PACKING_TRUNCATION=false
  export ABORT_ON_EXEC_TRUNCATION=false
  # 清掉可能污染的 AppWorld / lite 变量
  unset DATA_DIR_APPWORLD APPWORLD_PARTIAL_CREDIT PER_ROLLOUT_TIMEOUT || true
  export PER_ROLLOUT_TIMEOUT=600
fi

export METHOD="${METHOD:-lambda0}"
export RUN_ID="${RUN_ID:-trunc_probe_${METHOD}_$(date +%Y%m%d_%H%M%S)}"

echo "============================================================"
echo " Truncation probe"
echo "  MODEL=${MODEL_PATH}"
echo "  DATA=${DATA_DIR}"
echo "  GPUS=${NUM_GPUS}  BATCH×N=${BATCH_SIZE}×${ROLLOUT_N}  STEPS=${TOTAL_STEPS}"
echo "  → expect ~$((BATCH_SIZE * ROLLOUT_N * TOTAL_STEPS)) chains"
echo "  K_MAX_CAP=${K_MAX_CAP}"
echo "  RUN_ID=${RUN_ID}"
echo "============================================================"

bash "${SCRIPT_DIR}/run_stharc_triangle_diagnostic.sh"

STHARC_HOME="${STHARC_HOME:-/mnt/workspace/stharc}"
TRACE_DIR="${STHARC_HOME}/runs/public/rollout_logs/stharc_grpo_public_${RUN_ID}/traces"
echo ""
echo "============================================================"
echo " Truncation summary: ${TRACE_DIR}"
echo "============================================================"
if [[ -d "${TRACE_DIR}" ]]; then
  python3 "${SCRIPT_DIR}/summarize_truncation_traces.py" --trace-dir "${TRACE_DIR}" --limit 20
else
  echo "WARNING: traces not found at ${TRACE_DIR}"
  echo "Try: find ${STHARC_HOME}/runs -type d -name traces | tail"
fi
