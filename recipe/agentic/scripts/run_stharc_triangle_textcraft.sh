#!/usr/bin/env bash
# ======================================================================
# TextCraft 主表对照：λ=0 / STHARC / Random-PU / SAGE-style / SkillRise-style
# ======================================================================
# 所有任务保留上游奖励，不按读写类型屏蔽或整形；ExactMatch 设置不变。
# 此发布协议不等价于原实验的奖励协议。
#
# 用法（在 training_code_public 目录）：
#   METHOD=lambda0 MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#     bash recipe/agentic/scripts/run_stharc_triangle_textcraft.sh
#
#   METHOD=stharc STHARC_LAM=0.25 STHARC_ATTRIBUTION=latest_only \
#     bash recipe/agentic/scripts/run_stharc_triangle_textcraft.sh
#
#   METHOD=random_pu STHARC_LAM=0.25 \
#     bash recipe/agentic/scripts/run_stharc_triangle_textcraft.sh
#
#   METHOD=sage MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#     bash recipe/agentic/scripts/run_stharc_triangle_textcraft.sh
#
#   METHOD=skillrise MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#     bash recipe/agentic/scripts/run_stharc_triangle_textcraft.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:-stharc}"

if [[ "${ALLOW_PPO_SYNC:-0}" != "1" ]]; then
  export PPO_ENTRY=verl.trainer.main_ppo
fi
echo "[triangle-textcraft] PPO_ENTRY=${PPO_ENTRY:-verl.trainer.main_ppo}"

case "${METHOD}" in
  lambda0|lam0|chain_grpo)
    export STHARC_CREDIT=pu
    export STHARC_LAM=0
    export STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-gamma_decay}"
    export RUN_ID="${RUN_ID:-textcraft_mechlike_lambda0_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle-textcraft] METHOD=lambda0 → lam=0 (mechlike deferred_topo)"
    ;;
  stharc|correct|pu)
    export STHARC_CREDIT=pu
    if [[ "${FORCE_STHARC_LAM:-0}" == "1" || "${FORCE_STHARC_LAM:-0}" == "true" ]]; then
      export STHARC_LAM="${STHARC_LAM:-0.25}"
    else
      export STHARC_LAM=0.25
    fi
    export STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-latest_only}"
    export STHARC_PU_REQUIRE_CLEAN="${STHARC_PU_REQUIRE_CLEAN:-true}"
    export RUN_ID="${RUN_ID:-textcraft_mechlike_stharc_lo025_puclean_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle-textcraft] METHOD=stharc → lam=${STHARC_LAM} attr=${STHARC_ATTRIBUTION} pu_require_clean=${STHARC_PU_REQUIRE_CLEAN}"
    ;;
  random_pu|random|shuffle)
    export STHARC_CREDIT=pu
    if [[ "${FORCE_STHARC_LAM:-0}" == "1" || "${FORCE_STHARC_LAM:-0}" == "true" ]]; then
      export STHARC_LAM="${STHARC_LAM:-0.25}"
    else
      export STHARC_LAM=0.25
    fi
    export STHARC_ATTRIBUTION=random_shuffle
    export STHARC_SHUFFLE_SEED="${STHARC_SHUFFLE_SEED:-0}"
    export RUN_ID="${RUN_ID:-textcraft_mechlike_random_pu_025_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle-textcraft] METHOD=random_pu → lam=${STHARC_LAM} attr=random_shuffle"
    ;;
  sage|sage_style)
    export STHARC_CREDIT=sage
    export STHARC_LAM=0
    export STHARC_RUN_FROZEN=false
    export STHARC_PU_REQUIRE_CLEAN="${STHARC_PU_REQUIRE_CLEAN:-true}"
    export RUN_ID="${RUN_ID:-textcraft_mechlike_sage_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle-textcraft] METHOD=sage → credit=sage lam=0 frozen=off pu_require_clean=${STHARC_PU_REQUIRE_CLEAN}"
    ;;
  skillrise|skill_rise)
    export STHARC_CREDIT=skillrise
    export STHARC_LAM=0
    export STHARC_RUN_FROZEN=false
    export STHARC_SKILLRISE_GAMMA="${STHARC_SKILLRISE_GAMMA:-0.6}"
    export STHARC_PU_REQUIRE_CLEAN="${STHARC_PU_REQUIRE_CLEAN:-true}"
    export RUN_ID="${RUN_ID:-textcraft_mechlike_skillrise_$(date +%Y%m%d_%H%M%S)}"
    echo "[triangle-textcraft] METHOD=skillrise → credit=skillrise lam=0 frozen=off γ_curate=${STHARC_SKILLRISE_GAMMA} pu_require_clean=${STHARC_PU_REQUIRE_CLEAN}"
    ;;
  *)
    echo "ERROR: METHOD must be lambda0 | stharc | random_pu | sage | skillrise  (got: ${METHOD})"
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/run_stharc_grpo_textcraft.sh"
