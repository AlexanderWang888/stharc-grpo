#!/usr/bin/env bash
# ======================================================================
# Agentic GRPO 训练 — Gemma4 12B (unified) / ai_to_c
# ======================================================================
# 环境：conda activate RL12B（vLLM nightly + torch cu129 + verl）
#
# 用法（在 verl 目录下执行）：
#   cd /mnt/workspace/agentic_rl/rl/verl
#   bash recipe/agentic/scripts/run_ai_to_c_grpo_12b.sh
#
# 覆盖举例：
#   MODEL_PATH=/mnt/workspace/ckpt/gemma-4-12B-it TOTAL_STEPS=10 \
#     bash recipe/agentic/scripts/run_ai_to_c_grpo_12b.sh
# ======================================================================
set -euo pipefail

# ---------- 根目录 ----------
BASE_DIR="${BASE_DIR:-/mnt/workspace/agentic_rl/rl/runs}"

# ---------- 路径与基础参数 ----------
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/gemma-4-12B-it}"
AI2C_REPO_PATH="${AI2C_REPO_PATH:-/mnt/workspace/agentic_rl/rl/ai2c}"
DATA_DIR="${DATA_DIR:-/mnt/workspace/agentic_rl/rl/data/ai_to_c_2w}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/hydra/grpo_train_12b_$(date +%Y%m%d_%H%M%S)}"
SESSION_DIR="${SESSION_DIR:-${BASE_DIR}/sessions}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/checkpoints/grpo_gemma4_12b}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${BASE_DIR}/swanlab_logs}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"

# ---------- vLLM（agentic env 外部推理，colocate rollout 仍用 actor 权重） ----------
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-gemma-4-12B-it}"

# ---------- 规模参数 ----------
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"              # 12B rollout 单卡即可

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"   # 12B 显存充裕，不必像 31B 压到 0.35

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"

SAVE_FREQ="${SAVE_FREQ:-50}"
KEEP_CKPTS="${KEEP_CKPTS:-10}"
EVAL_FREQ="${EVAL_FREQ:-50}"

# batch —— 先跑通：尽量小；约束 train_batch_size * rollout.n 能被 n_gpus 整除
# 默认 4 * 2 = 8
BATCH_SIZE="${BATCH_SIZE:-4}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-4}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
LOG_PROB_MICRO="${LOG_PROB_MICRO:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"

PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-32768}"

# 序列长度 —— 12B 不必像 31B 那样压 ppo_max_token_len；先保持与 31B 脚本一致，稳定后可加大
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-14000}"
MAX_RESP_LEN="${MAX_RESP_LEN:-12000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-28000}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# ---------- 命令行覆盖 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)        MODEL_PATH="$2";                           shift 2 ;;
        --ai2c-repo-path)    AI2C_REPO_PATH="$2";                       shift 2 ;;
        --data-dir)          DATA_DIR="$2";                             shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";                           shift 2 ;;
        --base-dir)          BASE_DIR="$2";                             shift 2 ;;
        --vllm-base-url)     VLLM_BASE_URL="$2";                        shift 2 ;;
        --vllm-model-name)   VLLM_MODEL_NAME="$2";                      shift 2 ;;
        --num-gpus)          NUM_GPUS="$2";                             shift 2 ;;
        --epochs)            TOTAL_EPOCHS="$2";                         shift 2 ;;
        --steps)             TOTAL_STEPS="$2";                          shift 2 ;;
        --save-freq)         SAVE_FREQ="$2";                            shift 2 ;;
        --keep-ckpts)        KEEP_CKPTS="$2";                           shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; PPO_MINI_BATCH="$2";      shift 2 ;;
        --rollout-n)         ROLLOUT_N="$2";                            shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

mkdir -p "${SESSION_DIR}" "${CKPT_DIR}" "${SWANLAB_LOG_DIR}" "${LOG_DIR}"

export PYTHONPATH="${AI2C_REPO_PATH}:${PYTHONPATH:-}"
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR}"

LOG_FILE="${LOG_DIR}/grpo_train_12b_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo " Agentic GRPO 训练 — Gemma4 12B"
echo "========================================"
echo " Model:        ${MODEL_PATH}"
echo " ai2c repo:    ${AI2C_REPO_PATH}"
echo " vLLM:         ${VLLM_BASE_URL}  (${VLLM_MODEL_NAME})"
echo " Data:         ${DATA_DIR}"
echo " Output:       ${OUTPUT_DIR}"
echo " Checkpoints:  ${CKPT_DIR}  (keep=${KEEP_CKPTS}，freq=每 ${SAVE_FREQ} step)"
echo " SwanLab:      ${SWANLAB_LOG_DIR}"
echo " Sessions:     ${SESSION_DIR}"
echo " Log:          ${LOG_FILE}"
echo " GPUs:         ${NUM_GPUS}  |  TP: ${TP_SIZE}"
echo " Steps:        ${TOTAL_STEPS}  |  Epochs: ${TOTAL_EPOCHS}"
echo " BatchSize:    ${BATCH_SIZE}  |  MiniBS: ${PPO_MINI_BATCH}  |  MicroBS: ${PPO_MICRO_BATCH}"
echo " Rollout.n:    ${ROLLOUT_N}"
echo " SeqLen:       prompt=${MAX_PROMPT_LEN}  resp=${MAX_RESP_LEN}  model=${MAX_MODEL_LEN}"
echo " GPU MemUtil:  ${GPU_MEM_UTIL}  |  MaxNumSeqs: ${MAX_NUM_SEQS}"
echo " FSDP:         无 param/optimizer offload  |  entropy_coeff=0"
echo " DynBSZ:       ppo_max_token_len=${PPO_MAX_TOKEN_LEN}"
echo "========================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${EVAL_FREQ}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep="${KEEP_CKPTS}" \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=agentic_train \
    trainer.experiment_name=grpo_gemma4_12b_entropy0 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size=4 \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/agentic/config/agent_loop_list.yaml \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    \
    reward.reward_model.enable=false \
    \
    +agentic.env._target_=recipe.agentic.envs.AiToCReactEnv \
    "+agentic.env.ai_to_c_repo_path=${AI2C_REPO_PATH}" \
    "+agentic.env.session_dir_override=${SESSION_DIR}" \
    +agentic.env.record_subagent_traces=false \
    "+agentic.vllm_base_url=${VLLM_BASE_URL}" \
    "+agentic.vllm_model_name=${VLLM_MODEL_NAME}" \
    +agentic.per_rollout_timeout_sec=600 \
    +agentic.reward.normalize=true \
    +agentic.reward.clip_min=0.0 \
    +agentic.reward.clip_max=1.0 \
    "+agentic.reward.components=[{_target_: recipe.agentic.reward.components.TaskCompletedReward, weight: 1.0},{_target_: recipe.agentic.reward.components.ToolCallSuccessReward, weight: 0.3}]" \
    \
    ray_kwargs.ray_init.num_cpus=32 \
    2>&1 | tee "${LOG_FILE}"

echo "========================================"
echo " Training finished."
echo " Log:         ${LOG_FILE}"
echo " Checkpoints: ${CKPT_DIR}"
echo " SwanLab:     swanlab upload ${SWANLAB_LOG_DIR}"
echo "========================================"
