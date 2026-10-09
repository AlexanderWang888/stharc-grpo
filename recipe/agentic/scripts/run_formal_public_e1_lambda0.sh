#!/usr/bin/env bash
# ======================================================================
# 正式实验 E1：Public diagnostic 三角 — λ=0（Chain-GRPO baseline）
# 完整 smoke（不滤 K）· Qwen3.5-9B · 直接 GRPO · 截断只检测不硬拦
# ======================================================================
# 依赖：32k probe 已跑通（trunc_probe_32k_*）。OOM 时再降 MAX_RESP_LEN=4096。
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

if [[ "${ALLOW_ENV_OVERRIDE:-0}" != "1" ]]; then
  export MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B
  # 完整 smoke：mean K≈14.7，max K=32；不按 k12 过滤
  export DATA_DIR=data/stharc_public_smoke
  _NG="$(_detect_gpus)"
  if [[ -z "${_NG}" || "${_NG}" -lt 1 ]]; then
    _NG=8
  fi
  export NUM_GPUS="${_NG}"
  export TP_SIZE=1
  export GPU_MEM_UTIL=0.40
  export ROLLOUT_N=2
  export BATCH_SIZE=$(( NUM_GPUS / ROLLOUT_N ))
  if (( BATCH_SIZE < 1 )); then
    export BATCH_SIZE=1
    export ROLLOUT_N="${NUM_GPUS}"
  fi
  export NUM_AGENT_WORKERS=$((BATCH_SIZE * ROLLOUT_N))
  export VAL_BATCH_SIZE="${NUM_AGENT_WORKERS}"
  export TOTAL_STEPS=20
  export SAVE_FREQ=0
  export TEST_FREQ=0
  # 预算跟数据真实 max K；不要用 12 人为压预算
  export K_MAX_CAP=999
  export STRICT_CHAIN_BUDGET=0
  export MAX_PROMPT_LEN=4096
  # 探针上 4096 仍有个别 task_resp packing；正式抬到 8192（OOM 再退）
  export MAX_RESP_LEN=8192
  export MAX_TASK_CONTEXT_LEN=4096
  export VLLM_MAX_MODEL_LEN=32768
  export STHARC_MAX_STEPS=20
  unset MAX_MODEL_LEN CHAIN_RESP_LEN PPO_MAX_TOKEN_LEN K_MAX || true
  # 32k + 长 resp：并发收紧，避免显存炸
  export MAX_NUM_SEQS=4
  export STHARC_RUN_FROZEN=true
  export ABORT_ON_PACKING_TRUNCATION=false
  export ABORT_ON_EXEC_TRUNCATION=false
fi

export METHOD=lambda0
export RUN_ID="${RUN_ID:-formal_e1_lambda0_$(date +%Y%m%d_%H%M%S)}"

echo "============================================================"
echo " E1 Formal Public λ=0 (full smoke, no K filter)"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  DATA_DIR=${DATA_DIR}"
echo "  NUM_GPUS=${NUM_GPUS} BATCH×N=${BATCH_SIZE}×${ROLLOUT_N}"
echo "  MAX_RESP=${MAX_RESP_LEN} VLLM=${VLLM_MAX_MODEL_LEN} STEPS_ENV=${STHARC_MAX_STEPS}"
echo "  K_MAX_CAP=${K_MAX_CAP} STRICT=${STRICT_CHAIN_BUDGET}"
echo "  RUN_ID=${RUN_ID}"
echo "============================================================"

exec bash "${SCRIPT_DIR}/run_stharc_triangle_diagnostic.sh"
