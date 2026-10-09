#!/usr/bin/env bash
# ======================================================================
# Agentic GRPO 同步训练启动脚本 — ai_to_c 32B 主脑
# ======================================================================
# 前置条件：
#   1. 训练数据已通过 prepare_dataset.py 转为 parquet
#   2. vLLM 服务已在 VLLM_BASE_URL 上启动（默认 127.0.0.1:8000）
#   3. Ray 集群已就绪（或单机 ray start --head）
#
# 用法：
#   bash verl/recipe/agentic/scripts/run_ai_to_c_32b_sync.sh \
#       --vllm-base-url http://127.0.0.1:8000/v1 \
#       --model-path /data/models/Qwen2.5-32B-Instruct
# ======================================================================
set -euo pipefail

# ---------- 默认参数 ----------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-32B-Instruct}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen-32b-rl}"
DATA_DIR="${DATA_DIR:-data/ai_to_c}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/agentic_grpo_sync}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"            # tensor parallel for rollout
PP_SIZE="${PP_SIZE:-1}"            # pipeline parallel
TRAIN_STEPS="${TRAIN_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-50}"
WANDBB_PROJECT="${WANDBB_PROJECT:-agentic_train_offline}"
WANDBB_NAME="${WANDBB_NAME:-ai_to_c_32b_sync}"

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
CONFIG_PATH="${RECIPE_DIR}/config/agentic_grpo_sync.yaml"

echo "========================================"
echo " Agentic GRPO Sync — ai_to_c 32B"
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
# 保持 Hydra 默认 ppo_trainer 配置，通过 override 传入所有自定义字段
# + 前缀表示新增字段（不在 ppo_trainer.yaml 中定义的）
python -m verl.trainer.main_ppo \
    hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.pipeline_model_parallel_size="${PP_SIZE}" \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.prompt_length=16384 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${CONFIG_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.reward_model.enable=false \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=16384 \
    data.max_response_length=8192 \
    +data.return_raw_prompt=true \
    data.return_raw_chat=true \
    data.shuffle=true \
    data.filter_overlong_prompts=true \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TRAIN_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${EVAL_FREQ}" \
    trainer.project_name=agentic_train_offline \
    trainer.experiment_name=ai_to_c_32b_sync \
    trainer.logger='["console","wandb"]' \
    +trainer.wandb.project="${WANDBB_PROJECT}" \
    +trainer.wandb.name="${WANDBB_NAME}" \
    +data.data_source=ai_to_c \
    +agentic.env._target_=recipe.agentic.envs.AiToCReactEnv \
    +agentic.env.ai_to_c_repo_path=/workspace/ai2c \
    +agentic.env.session_dir_override=/tmp/rl_sessions \
    +agentic.env.record_subagent_traces=false \
    +agentic.reward.normalize=true \
    +agentic.reward.clip_min=0.0 \
    +agentic.reward.clip_max=1.0 \
    +agentic.vllm_base_url="${VLLM_BASE_URL}" \
    +agentic.vllm_model_name="${VLLM_MODEL_NAME}" \
    +agentic.per_rollout_timeout_sec=600 \
    +agentic.agent_loop_config_path="${CONFIG_PATH}"

echo "Training finished. Checkpoints in: ${OUTPUT_DIR}"