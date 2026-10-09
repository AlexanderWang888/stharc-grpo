#!/usr/bin/env bash
# ======================================================================
# Agentic GRPO 全异步训练启动脚本 — ai_to_c 32B 主脑
# ======================================================================
# 与 sync 版的区别：
#   - rollout 使用 vllm_async engine，n=4 并行采样
#   - train_batch_size=256, ppo_mini_batch_size=256
#   - AgenticAgentLoop 代码零修改，仅配置不同
#
# 前置条件同 sync 版
# ======================================================================
set -euo pipefail

# ---------- 默认参数 ----------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen-32b-rl}"
DATA_DIR="${DATA_DIR:-data/ai_to_c}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/agentic_grpo_async}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-50}"
WANDBB_PROJECT="${WANDBB_PROJECT:-agentic_train_offline}"
WANDBB_NAME="${WANDBB_NAME:-ai_to_c_32b_async}"

# ---------- 解析命令行覆盖 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)       MODEL_PATH="$2";       shift 2 ;;
        --vllm-base-url)    VLLM_BASE_URL="$2";    shift 2 ;;
        --vllm-model-name)  VLLM_MODEL_NAME="$2";  shift 2 ;;
        --data-dir)         DATA_DIR="$2";         shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";       shift 2 ;;
        --num-gpus)         NUM_GPUS="$2";         shift 2 ;;
        --train-steps)      TRAIN_STEPS="$2";      shift 2 ;;
        --save-freq)        SAVE_FREQ="$2";        shift 2 ;;
        --eval-freq)        EVAL_FREQ="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${RECIPE_DIR}/config/agentic_grpo_async.yaml"

echo "========================================"
echo " Agentic GRPO Async — ai_to_c 32B"
echo "========================================"
echo " Model:        ${MODEL_PATH}"
echo " vLLM:         ${VLLM_BASE_URL} (${VLLM_MODEL_NAME})"
echo " Data:         ${DATA_DIR}"
echo " Output:       ${OUTPUT_DIR}"
echo " GPUs:         ${NUM_GPUS}"
echo " Steps:        ${TRAIN_STEPS}"
echo " Config:       ${CONFIG_PATH}"
echo "========================================"

# ---------- 启动训练 ----------
python -m verl.trainer.main_ppo \
    hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    --config-path "${RECIPE_DIR}/config" \
    --config-name agentic_grpo_async \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.pipeline_model_parallel_size="${PP_SIZE}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${CONFIG_PATH}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    trainer.total_training_steps="${TRAIN_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${EVAL_FREQ}" \
    trainer.wandb.project="${WANDBB_PROJECT}" \
    trainer.wandb.name="${WANDBB_NAME}" \
    agentic.vllm_base_url="${VLLM_BASE_URL}" \
    agentic.vllm_model_name="${VLLM_MODEL_NAME}" \
    agentic.agent_loop_config_path="${CONFIG_PATH}"

echo "Training finished. Checkpoints in: ${OUTPUT_DIR}"