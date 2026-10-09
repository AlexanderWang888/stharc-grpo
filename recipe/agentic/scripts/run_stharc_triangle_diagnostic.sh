#!/usr/bin/env bash
# ======================================================================
# Diagnostic 三角短训：λ=0 / STHARC / Random-PU（M2 + M2b / E2 + E4）
# ======================================================================
# 用法（在 training_code_public 目录）：
#   METHOD=lambda0|stharc|random_pu \
#   NUM_GPUS=2 TP_SIZE=1 GPU_MEM_UTIL=0.5 \
#   BATCH_SIZE=4 ROLLOUT_N=2 TOTAL_STEPS=20 SAVE_FREQ=0 \
#   MAX_PROMPT_LEN=4096 MAX_RESP_LEN=2048 MAX_TASK_CONTEXT_LEN=512 \
#   DATA_DIR=data/stharc_public_smoke \
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#   bash recipe/agentic/scripts/run_stharc_triangle_diagnostic.sh
#
# 注意：
# - MAX_RESP_LEN 是单任务预算；整链由 public 脚本按 K_MAX 自动算 CHAIN_RESP_LEN
# - public 默认 PPO_ENTRY=verl.trainer.main_ppo（勿再切回 main_ppo_sync，会 .offsets() 崩）
# - 截断默认只检测：SwanLab trunc/* + trace.truncation.truncated_task_idxs
#   汇总：python3 recipe/agentic/scripts/summarize_truncation_traces.py --trace-dir ...
#
# 建议连跑三次（同一 DATA/MODEL/步数，只改 METHOD）：
#   for m in lambda0 stharc random_pu; do METHOD=$m ... bash $0; done
#
# 期望（diagnostic 上）：
#   STHARC (正确 PU) ≳ λ=0 ≳ Random-PU   或至少 Random-PU 不明显优于 STHARC
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:-stharc}"

# 强制 main_ppo（忽略环境里残留的 PPO_ENTRY=main_ppo_sync）。
# 若确需 sync：ALLOW_PPO_SYNC=1 PPO_ENTRY=verl.trainer.main_ppo_sync ...
if [[ "${ALLOW_PPO_SYNC:-0}" != "1" ]]; then
  export PPO_ENTRY=verl.trainer.main_ppo
fi
echo "[triangle] PPO_ENTRY=${PPO_ENTRY}"

case "${METHOD}" in
  lambda0|lam0|chain_grpo)
    export STHARC_LAM=0
    export STHARC_ATTRIBUTION=gamma_decay
    export RUN_ID="${RUN_ID:-triangle_lambda0_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle] METHOD=lambda0  → lam=0, attribution=gamma_decay"
    ;;
  stharc|correct|pu)
    export STHARC_LAM="${STHARC_LAM:-0.5}"
    # 允许外层覆盖（如 mech_v10: STHARC_ATTRIBUTION=latest_only）
    export STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-gamma_decay}"
    export RUN_ID="${RUN_ID:-triangle_stharc_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle] METHOD=stharc   → lam=${STHARC_LAM}, attribution=${STHARC_ATTRIBUTION}"
    ;;
  random_pu|random|shuffle)
    export STHARC_LAM="${STHARC_LAM:-0.5}"
    export STHARC_ATTRIBUTION=random_shuffle
    export STHARC_SHUFFLE_SEED="${STHARC_SHUFFLE_SEED:-0}"
    export RUN_ID="${RUN_ID:-triangle_random_pu_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle] METHOD=random_pu → lam=${STHARC_LAM}, attribution=random_shuffle, seed=${STHARC_SHUFFLE_SEED}"
    ;;
  *)
    echo "ERROR: METHOD must be lambda0 | stharc | random_pu  (got: ${METHOD})"
    exit 2
    ;;
esac

# 三角短训默认：不存大 ckpt；开 val；可被外层覆盖
export SAVE_FREQ="${SAVE_FREQ:-0}"
export TOTAL_STEPS="${TOTAL_STEPS:-70}"
export TEST_FREQ="${TEST_FREQ:-10}"

exec bash "${SCRIPT_DIR}/run_stharc_grpo_public.sh"
