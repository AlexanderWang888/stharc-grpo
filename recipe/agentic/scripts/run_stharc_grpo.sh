#!/usr/bin/env bash
# ======================================================================
# STHARC-GRPO 训练启动脚本 — Qwen2.5-32B / ai_to_c
# ======================================================================
# STHARC = Skill & Task Harness Agentic RL with Chains
# 核心思路：用 task chain（K 个顺序任务）替代单任务 rollout，
#          通过 prospective_bonus 把后续任务收益归因到前面的 skill_manage 操作。
#
# K=3，G=4；STHARC_* 默认见 _research_workspace_paths.sh（含 RUN_FROZEN）
#
# 用法（在 training_code_public 目录下执行）：
#   cd ${WS}/research/training_code_public
#   bash recipe/agentic/scripts/run_stharc_grpo.sh
#
# smoke test（2卡，快速验证流程）：
#   NUM_GPUS=2 TP_SIZE=2 BATCH_SIZE=2 ROLLOUT_N=2 TOTAL_STEPS=5 \
#   K=3 G=2 STHARC_LAM=0.5 \
#   MAX_PROMPT_LEN=8192 MAX_RESP_LEN=8192 PPO_MAX_TOKEN_LEN=10000 \
#   MAX_MODEL_LEN=16384 GPU_MEM_UTIL=0.35 \
#   bash recipe/agentic/scripts/run_stharc_grpo.sh
#
# 数据集准备（先于训练运行；--G 1，GRPO 的 G 由 rollout.n 控制）：
#   PYTHONPATH=.:verl python -m recipe.agentic.data.prepare_dataset \\
#       --train-input ../filtered_traces.jsonl \\
#       --train-output ../data/stharc_2w/train.parquet \\
#       --K 3 --G 1 --strategy traces
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_research_workspace_paths.sh"

# ---------- 根目录 ----------
BASE_DIR="${BASE_DIR}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# ---------- 路径与基础参数 ----------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}"
# AI2C_REPO_PATH / DATA_DIR 由 _research_workspace_paths.sh 提供

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/hydra/stharc_grpo_${RUN_ID}}"
SESSION_DIR="${SESSION_DIR:-${BASE_DIR}/sessions}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/checkpoints/stharc_grpo_${RUN_ID}}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${BASE_DIR}/swanlab_logs/stharc_grpo_${RUN_ID}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${BASE_DIR}/rollout_logs/stharc_grpo_${RUN_ID}}"
AGENT_LOOP_LIST_PATH="${AGENT_LOOP_LIST_PATH:-recipe/agentic/config/agent_loop_list.yaml}"

# ---------- vLLM ----------
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen-32b-rl}"

# ---------- 规模参数 ----------
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"

BATCH_SIZE="${BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"          # G = GRPO group size
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-16384}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"
PER_ROLLOUT_TIMEOUT="${PER_ROLLOUT_TIMEOUT:-600}"

# ---------- STHARC 专有参数 ----------
K="${K:-3}"                          # chain 长度
G="${G:-4}"                          # GRPO group size
# STHARC_GAMMA / LAM / ATTRIBUTION / RUN_FROZEN / BASE_USER_ID
# 已由 _research_workspace_paths.sh 设定（可 env 覆盖）

# ---------- 目录准备 ----------
mkdir -p "${OUTPUT_DIR}" "${SESSION_DIR}" "${CKPT_DIR}" "${SWANLAB_LOG_DIR}" \
         "${ROLLOUT_DATA_DIR}" "${BASE_DIR}/logs"

# ---------- 数据路径验证 ----------
if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "[STHARC] WARNING: train.parquet not found at ${DATA_DIR}/train.parquet"
    echo "[STHARC] 请先运行数据准备脚本："
    echo "  PYTHONPATH=.:verl python -m recipe.agentic.data.prepare_dataset \\"
    echo "      --train-input ${RESEARCH_ROOT}/filtered_traces.jsonl \\"
    echo "      --train-output ${DATA_DIR}/train.parquet \\"
    echo "      --K ${K} --G 1 --strategy traces"
fi

echo "============================================================"
echo " STHARC-GRPO 启动参数"
echo "  MODEL_PATH       : ${MODEL_PATH}"
echo "  DATA_DIR         : ${DATA_DIR}"
echo "  CKPT_DIR         : ${CKPT_DIR}"
echo "  NUM_GPUS/TP_SIZE : ${NUM_GPUS} / ${TP_SIZE}"
echo "  BATCH/ROLLOUT_N  : ${BATCH_SIZE} / ${ROLLOUT_N}"
echo "  K / G            : ${K} / ${G}"
echo "  GAMMA / LAM      : ${STHARC_GAMMA} / ${STHARC_LAM}"
echo "  ATTRIBUTION      : ${STHARC_ATTRIBUTION}"
echo "  RUN_FROZEN       : ${STHARC_RUN_FROZEN}"
echo "  TOTAL_STEPS      : ${TOTAL_STEPS}"
echo "============================================================"

# ---------- 启动训练 ----------
# 与生产 / public 一致走 main_ppo，避开 main_ppo_sync+TransferQueue densify 崩溃
python -m verl.trainer.main_ppo \
    --config-path "$(pwd)/recipe/agentic/config" \
    --config-name stharc_grpo \
    hydra.run.dir="${OUTPUT_DIR}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESP_LEN}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_LIST_PATH}" \
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}" \
    "actor_rollout_ref.rollout.vllm.max_model_len=${MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.vllm.max_num_seqs=${MAX_NUM_SEQS}" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}" \
    "actor_rollout_ref.actor.micro_batch_size_per_gpu=2" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size=32 \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESP_LEN}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    "trainer.default_hdfs_dir=${CKPT_DIR}" \
    "trainer.logger=[console,swanlab]" \
    "trainer.swanlab.save_dir=${SWANLAB_LOG_DIR}" \
    stharc.env._target_=recipe.agentic.envs.AiToCReactEnv \
    "stharc.env.ai_to_c_repo_path=${AI2C_REPO_PATH}" \
    "stharc.env.session_dir_override=${SESSION_DIR}" \
    "stharc.vllm_base_url=${VLLM_BASE_URL}" \
    "stharc.vllm_model_name=${VLLM_MODEL_NAME}" \
    "stharc.per_rollout_timeout_sec=${PER_ROLLOUT_TIMEOUT}" \
    "stharc.gamma=${STHARC_GAMMA}" \
    "stharc.lam=${STHARC_LAM}" \
    "stharc.attribution=${STHARC_ATTRIBUTION}" \
    "stharc.run_frozen_chain=${STHARC_RUN_FROZEN}" \
    "stharc.base_user_id=${STHARC_BASE_USER_ID}" \
    2>&1 | tee "${BASE_DIR}/logs/stharc_grpo_${RUN_ID}.log"
