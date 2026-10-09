#!/usr/bin/env bash
# research/training_code_public 路径约定（仅用于 research 实验，不改动主仓 verl/）。
#
#   ${WS_ROOT}/
#     ai2c/                         共用业务代码
#     agentic_train_offline/        共用 bridge / 评测
#     research/
#       training_code_public/              本仓库
#       data/stharc_2w/             STHARC parquet（--G 1，靠 rollout.n 扩采样）
#       runs/                       本实验产物（可选）
#
# 由 recipe/agentic/scripts/*.sh source。

_AGENTIC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${0}}")" && pwd)"
# scripts → agentic → recipe → 当前副本根；不依赖 checkout 的目录名
VERL_RESEARCH_DIR="$(cd "${_AGENTIC_SCRIPT_DIR}/../../.." && pwd)"
RESEARCH_ROOT="$(cd "${VERL_RESEARCH_DIR}/.." && pwd)"
WS_ROOT="$(cd "${RESEARCH_ROOT}/.." && pwd)"

export RESEARCH_ROOT WS_ROOT VERL_RESEARCH_DIR
export AI2C_REPO_PATH="${AI2C_REPO_PATH:-${WS_ROOT}/ai2c}"
export SERVER_ROOT="${SERVER_ROOT:-${RESEARCH_ROOT}}"

# STHARC 数据：仅用 --G 1 的 stharc_2w（G 由 trainer rollout.n 控制）。
# 不再回退 data/stharc（旧版 parquet 内已复制 G 行，会与 rollout.n 重复扩采样）。
export DATA_DIR="${DATA_DIR:-${RESEARCH_ROOT}/data/stharc_2w}"

# 实验产物默认落在 research/runs，避免与主线 ${WS}/runs 混用
export BASE_DIR="${BASE_DIR:-${RESEARCH_ROOT}/runs}"

# ── STHARC 算法默认（唯一维护处；各 run_stharc_*.sh source 后直接用）────────
# 覆盖方式：启动前 export，例如 STHARC_RUN_FROZEN=false bash run_stharc_qwen08b_8gpu.sh
export STHARC_GAMMA="${STHARC_GAMMA:-0.9}"
export STHARC_LAM="${STHARC_LAM:-0.25}"
export STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-latest_only}"
export STHARC_RUN_FROZEN="${STHARC_RUN_FROZEN:-false}"
export STHARC_BASE_USER_ID="${STHARC_BASE_USER_ID:-default}"

# PYTHONPATH：training_code_public 必须最前，避免命中 ${WS_ROOT}/verl/recipe（主仓无 STHARC）
# 顺序：当前副本根 → 副本内 verl → ai2c → WS 根（agentic_train_offline）
_PY_PARTS=("${VERL_RESEARCH_DIR}")
if [[ -d "${VERL_RESEARCH_DIR}/verl" ]]; then
  _PY_PARTS+=("${VERL_RESEARCH_DIR}/verl")
fi
_PY_PARTS+=("${AI2C_REPO_PATH}" "${WS_ROOT}")
# 过滤旧 PYTHONPATH 里的主仓 ${WS_ROOT}/verl，防止 Ray worker 仍被污染
_FILTERED_OLD=""
if [[ -n "${PYTHONPATH:-}" ]]; then
  _WS_VERL="$(cd "${WS_ROOT}/verl" 2>/dev/null && pwd || true)"
  IFS=':' read -r -a _OLD_PARTS <<< "${PYTHONPATH}"
  for _p in "${_OLD_PARTS[@]}"; do
    [[ -z "${_p}" ]] && continue
    if [[ -n "${_WS_VERL}" && "${_p}" == "${_WS_VERL}" ]]; then
      continue
    fi
    _FILTERED_OLD="${_FILTERED_OLD:+${_FILTERED_OLD}:}${_p}"
  done
fi
export PYTHONPATH="$(IFS=:; echo "${_PY_PARTS[*]}")${_FILTERED_OLD:+:${_FILTERED_OLD}}"
