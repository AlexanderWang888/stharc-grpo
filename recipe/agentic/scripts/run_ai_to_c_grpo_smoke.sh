#!/usr/bin/env bash
# ======================================================================
# Agentic GRPO 同步训练 — Gemma4 31B
# ======================================================================
# 前置条件：
#   1. 训练数据已生成到 DATA_DIR（train.parquet / val.parquet）
#   2. vLLM 服务已启动（默认 http://127.0.0.1:8000/v1）
#   3. 模型路径正确（默认 /mnt/workspace/agentic_rl/checkpoints2）
#
# 用法（在 verl 目录下执行）：
#   cd /mnt/workspace/agentic_rl/rl/verl
#   bash recipe/agentic/scripts/run_ai_to_c_grpo_smoke.sh
#
# 覆盖参数举例：
#   DATA_DIR=/other/data OUTPUT_DIR=outputs/run2 BATCH_SIZE=16 \
#     bash recipe/agentic/scripts/run_ai_to_c_grpo_smoke.sh
# ======================================================================
set -euo pipefail

# ---------- 路径与基础参数 ----------
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/agentic_rl/checkpoints2}"
AI2C_REPO_PATH="${AI2C_REPO_PATH:-/mnt/workspace/agentic_rl/rl/ai2c}"
DATA_DIR="${DATA_DIR:-/mnt/workspace/agentic_rl/rl/data/ai_to_c_2k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_2k_1epoch}"
SESSION_DIR="${SESSION_DIR:-/tmp/rl_sessions}"

# ---------- vLLM ----------
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-gemma-4-31b-rl}"

# ---------- 规模参数 ----------
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-4}"              # rollout tensor parallel
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}" # 0.82 OOM on wake_up; 0.70留出训练阶段显存空间
# 2k数据，batch=16，1epoch ≈ 1900/16 ≈ 119步；设150作为上限，total_epochs=1自然截止
TRAIN_STEPS="${TRAIN_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-50}"

# batch size 系列
BATCH_SIZE="${BATCH_SIZE:-16}"                  # 必须是 n_gpus(8) 的倍数，从8提到16提高吞吐
PPO_MINI_BATCH="${PPO_MINI_BATCH:-16}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-2}"         # 从1提到2，提升training阶段GPU利用率
LOG_PROB_MICRO="${LOG_PROB_MICRO:-2}"           # 从1提到2

# 序列长度
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"        # 业务query不长，2048足够
MAX_RESP_LEN="${MAX_RESP_LEN:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"          # prompt(2048)+resp(2048)+buffer
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"              # batch(16)*n(8)=128，设64作为vLLM并发上限

# ---------- 命令行覆盖 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --ai2c-repo-path)    AI2C_REPO_PATH="$2";    shift 2 ;;
        --data-dir)          DATA_DIR="$2";           shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";         shift 2 ;;
        --vllm-base-url)     VLLM_BASE_URL="$2";      shift 2 ;;
        --vllm-model-name)   VLLM_MODEL_NAME="$2";    shift 2 ;;
        --num-gpus)          NUM_GPUS="$2";            shift 2 ;;
        --train-steps)       TRAIN_STEPS="$2";         shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; PPO_MINI_BATCH="$2"; shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

# ---------- PYTHONPATH ----------
export PYTHONPATH="${AI2C_REPO_PATH}:${PYTHONPATH:-}"

echo "========================================"
echo " Agentic GRPO — Gemma4 31B"
echo "========================================"
echo " Model:        ${MODEL_PATH}"
echo " ai2c repo:    ${AI2C_REPO_PATH}"
echo " vLLM:         ${VLLM_BASE_URL}  (${VLLM_MODEL_NAME})"
echo " Data:         ${DATA_DIR}"
echo " Output:       ${OUTPUT_DIR}"
echo " GPUs:         ${NUM_GPUS}  |  TP: ${TP_SIZE}"
echo " Steps:        ${TRAIN_STEPS}  |  BatchSize: ${BATCH_SIZE}"
echo " SeqLen:       prompt=${MAX_PROMPT_LEN}  resp=${MAX_RESP_LEN}"
echo "========================================"

# ---------- 启动训练 ----------
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TRAIN_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${EVAL_FREQ}" \
    trainer.logger=console \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESP_LEN}" \
    data.trust_remote_code=True \
    data.return_raw_chat=true \
    data.shuffle=true \
    data.filter_overlong_prompts=true \
    +data.data_source=ai_to_c \
    +data.return_raw_prompt=true \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO}" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    reward.reward_model.enable=false \
    \
    +agentic.env._target_=recipe.agentic.envs.AiToCReactEnv \
    "+agentic.env.ai_to_c_repo_path=${AI2C_REPO_PATH}" \
    +agentic.env.session_dir_override="${SESSION_DIR}" \
    +agentic.env.record_subagent_traces=false \
    "+agentic.vllm_base_url=${VLLM_BASE_URL}" \
    "+agentic.vllm_model_name=${VLLM_MODEL_NAME}" \
    +agentic.per_rollout_timeout_sec=300 \
    +agentic.reward.normalize=true \
    +agentic.reward.clip_min=0.0 \
    +agentic.reward.clip_max=1.0 \
    \
    ray_kwargs.ray_init.num_cpus=32 \
    2>&1 | tee grpo_2k.log

echo "========================================"
echo " Training finished. Log: $(pwd)/grpo_2k.log"
echo "========================================"
