#!/usr/bin/env bash
# ======================================================================
# STHARC-GRPO 训练启动脚本 — MuSiQue（真实硬依赖 QA / ExactMatch）
# ======================================================================
# 与 AppWorld / diagnostic 并列的数据集入口。底层复用 public 训练环
# （GenericBenchmarkEnv + ExactMatchReward），但默认数据、产物目录、主设定
# 与 musique verbatim short 对齐，避免再混用 stharc_public*。
#
# 数据准备（本机 research/ · 主表 mechlike n1k，write 前缀 + read 后缀）：
#   python3 dataset_explore/idea2_datasets/build_musique_verbatim_short.py \
#     --topology deferred_topo \
#     --out-dir dataset_explore/idea2_datasets_out/_musique_verbatim_mechlike_n1k \
#     --parquet-dir training_code_public/data/stharc_musique_verbatim_mechlike_n1k \
#     --n 1000 --val-ratio 0.1 --min-k 6 --max-k 10 --seed 42
#
# 用法（在 training_code_public 目录）：
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#   bash recipe/agentic/scripts/run_stharc_grpo_musique.sh
#
# 主表 / 对照臂：
#   METHOD=lambda0|stharc|sage|skillrise \
#     bash recipe/agentic/scripts/run_stharc_triangle_musique.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------- MuSiQue 路径（默认强制；防 shell 里残留 AppWorld DATA_DIR）----------
# 若曾 export DATA_DIR=.../stharc_appworld*，${DATA_DIR:-default} 会静默跑错数据集。
# 默认：deferred_topo mechlike n1k；旧 interleave：ALLOW_ENV_OVERRIDE=1 DATA_DIR=.../verbatim_short_n1k
_MUSIQUE_DATA_DEFAULT="${VERL_RESEARCH_DIR}/data/stharc_musique_verbatim_mechlike_n1k"
_MUSIQUE_BASE_DEFAULT="${STHARC_HOME:-/mnt/workspace/stharc}/runs/musique"
export STHARC_HOME="${STHARC_HOME:-/mnt/workspace/stharc}"
if [[ "${ALLOW_ENV_OVERRIDE:-0}" == "1" || "${ALLOW_ENV_OVERRIDE:-0}" == "true" ]]; then
  export DATA_DIR="${DATA_DIR:-${_MUSIQUE_DATA_DEFAULT}}"
  export BASE_DIR="${BASE_DIR:-${_MUSIQUE_BASE_DEFAULT}}"
else
  if [[ -n "${DATA_DIR:-}" && "${DATA_DIR}" != "${_MUSIQUE_DATA_DEFAULT}" ]]; then
    echo "[STHARC-musique] WARNING: ignoring leftover DATA_DIR=${DATA_DIR} (use ALLOW_ENV_OVERRIDE=1 to keep it)"
  fi
  export DATA_DIR="${_MUSIQUE_DATA_DEFAULT}"
  export BASE_DIR="${_MUSIQUE_BASE_DEFAULT}"
fi
case "${DATA_DIR}" in
  *appworld*|*diagnostic*)
    echo "[STHARC-musique] ERROR: DATA_DIR looks non-musique: ${DATA_DIR}" >&2
    echo "[STHARC-musique]        refuse to mix datasets. unset DATA_DIR or point to stharc_musique_*" >&2
    exit 2
    ;;
esac

# 主设定（与 diagnostic short / AppWorld 锁定一致）
export STHARC_RUN_FROZEN="${STHARC_RUN_FROZEN:-false}"
export STHARC_GAMMA="${STHARC_GAMMA:-0.9}"
export STHARC_LAM="${STHARC_LAM:-0.25}"
export STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-latest_only}"
export STHARC_FORMAT_FEW_SHOT="${STHARC_FORMAT_FEW_SHOT:-1}"
# 所有任务保留上游奖励，不按读写类型屏蔽或整形；ExactMatch 设置不变。
# 此发布协议不等价于原实验的奖励协议。
# 方案 A：脏链不上 PU（trunc / no_final / env fail → pb=0, chain_reward=total）
export STHARC_PU_REQUIRE_CLEAN="${STHARC_PU_REQUIRE_CLEAN:-true}"

# 满预算（禁止擅自改小；OOM 只动 batch / MEM / MAX_NUM_SEQS）
# 与 DATA_DIR 同理：shell 残留 MAX_RESP_LEN=1024 等会静默压预算 → packing 全截。
# 默认强制主表满预算；确需改：ALLOW_ENV_OVERRIDE=1 MAX_RESP_LEN=...
_MUSIQUE_PROMPT_DEFAULT=4096
_MUSIQUE_RESP_DEFAULT=8192
_MUSIQUE_CTX_DEFAULT=4096
_MUSIQUE_VLLM_DEFAULT=32768
_MUSIQUE_STEPS_DEFAULT=20
_MUSIQUE_KCAP_DEFAULT=999
if [[ "${ALLOW_ENV_OVERRIDE:-0}" == "1" || "${ALLOW_ENV_OVERRIDE:-0}" == "true" ]]; then
  export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-${_MUSIQUE_PROMPT_DEFAULT}}"
  export MAX_RESP_LEN="${MAX_RESP_LEN:-${_MUSIQUE_RESP_DEFAULT}}"
  export MAX_TASK_CONTEXT_LEN="${MAX_TASK_CONTEXT_LEN:-${_MUSIQUE_CTX_DEFAULT}}"
  export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_MUSIQUE_VLLM_DEFAULT}}"
  export STHARC_MAX_STEPS="${STHARC_MAX_STEPS:-${_MUSIQUE_STEPS_DEFAULT}}"
  export K_MAX_CAP="${K_MAX_CAP:-${_MUSIQUE_KCAP_DEFAULT}}"
else
  for _kv in \
    "MAX_PROMPT_LEN=${_MUSIQUE_PROMPT_DEFAULT}" \
    "MAX_RESP_LEN=${_MUSIQUE_RESP_DEFAULT}" \
    "MAX_TASK_CONTEXT_LEN=${_MUSIQUE_CTX_DEFAULT}" \
    "VLLM_MAX_MODEL_LEN=${_MUSIQUE_VLLM_DEFAULT}" \
    "STHARC_MAX_STEPS=${_MUSIQUE_STEPS_DEFAULT}" \
    "K_MAX_CAP=${_MUSIQUE_KCAP_DEFAULT}"; do
    _k="${_kv%%=*}"
    _v="${_kv#*=}"
    _cur="${!_k:-}"
    if [[ -n "${_cur}" && "${_cur}" != "${_v}" ]]; then
      echo "[STHARC-musique] WARNING: ignoring leftover ${_k}=${_cur} (force ${_v}; ALLOW_ENV_OVERRIDE=1 to keep)"
    fi
    export "${_k}=${_v}"
  done
  # CHAIN_RESP_LEN / PPO_MAX_TOKEN_LEN 若残留也会压整链；默认清掉让 public 脚本按 K 重算
  if [[ -n "${CHAIN_RESP_LEN:-}" ]]; then
    echo "[STHARC-musique] WARNING: unsetting leftover CHAIN_RESP_LEN=${CHAIN_RESP_LEN}"
    unset CHAIN_RESP_LEN
  fi
  if [[ -n "${PPO_MAX_TOKEN_LEN:-}" ]]; then
    echo "[STHARC-musique] WARNING: unsetting leftover PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN}"
    unset PPO_MAX_TOKEN_LEN
  fi
  if [[ -n "${K_MAX:-}" ]]; then
    echo "[STHARC-musique] WARNING: unsetting leftover K_MAX=${K_MAX} (re-scan parquet)"
    unset K_MAX
  fi
fi
export STRICT_CHAIN_BUDGET="${STRICT_CHAIN_BUDGET:-0}"
echo "[STHARC-musique] budget: PROMPT=${MAX_PROMPT_LEN} RESP=${MAX_RESP_LEN} CTX=${MAX_TASK_CONTEXT_LEN} VLLM=${VLLM_MAX_MODEL_LEN} STEPS=${STHARC_MAX_STEPS} K_CAP=${K_MAX_CAP}"

# 9B 常用起步规模（可覆盖）
export MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/Qwen3.5-9B}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export TOTAL_STEPS="${TOTAL_STEPS:-100}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
export ACTOR_LR="${ACTOR_LR:-5e-7}"

RUN_ID="${RUN_ID:-musique_$(date +%Y%m%d_%H%M%S)}"
export RUN_ID

if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
  echo "[STHARC-musique] ERROR: missing ${DATA_DIR}/train.parquet"
  echo "[STHARC-musique] build or rsync stharc_musique_verbatim_short first."
  exit 2
fi

echo "[STHARC-musique] DATA_DIR=${DATA_DIR}"
echo "[STHARC-musique] BASE_DIR=${BASE_DIR}"
echo "[STHARC-musique] MODEL_PATH=${MODEL_PATH}"
echo "[STHARC-musique] RUN_ID=${RUN_ID}  credit=${STHARC_CREDIT:-pu} lam=${STHARC_LAM} attr=${STHARC_ATTRIBUTION} frozen=${STHARC_RUN_FROZEN} pu_require_clean=${STHARC_PU_REQUIRE_CLEAN}"

exec bash "${SCRIPT_DIR}/run_stharc_grpo_public.sh"
