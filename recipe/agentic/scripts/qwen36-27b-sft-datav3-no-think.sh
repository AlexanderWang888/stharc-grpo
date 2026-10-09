#!/usr/bin/env bash
# ======================================================================
# Agentic GRPO 正式训练 — Qwen3.6 27B (SFT datav3 no_think) / ai_to_c
# 方案：fsdp1 + free_cache_engine=true（修复 backward OOM）
# ======================================================================
# 与 qwen36-27b.sh 的唯一差异：
#   MODEL_PATH 指向 SFT checkpoint：
#   /mnt/workspace/agentic_rl/agentic_sft/sft_checkpoints/qwen3_6_sft_datav3_no_think
# ======================================================================
# 用法（在 verl 目录下执行）：
#   cd /mnt/workspace/agentic_rl/rl/verl
#   DATA_DIR=/mnt/workspace/agentic_rl/rl/data/ai_to_c_2w TOTAL_STEPS=500 \
#     bash recipe/agentic/scripts/qwen36-27b-sft-datav3-no-think.sh
#
# 模型路径 / vLLM 服务名按实际环境覆盖：
#   MODEL_PATH=/mnt/workspace/agentic_rl/agentic_sft/sft_checkpoints/qwen3_6_sft_datav3_no_think \
#     VLLM_MODEL_NAME=qwen3.6-27b-rl \
#     bash recipe/agentic/scripts/qwen36-27b-sft-datav3-no-think.sh
#
# 2 卡临时 smoke（不改脚本默认值，跑通后仍用 8 卡正式训）：
#   NUM_GPUS=2 TP_SIZE=2 BATCH_SIZE=2 ROLLOUT_N=2 TOTAL_STEPS=5 \
#   MAX_PROMPT_LEN=8192 MAX_RESP_LEN=8192 PPO_MAX_TOKEN_LEN=10000 \
#   MAX_MODEL_LEN=16384 MAX_NUM_SEQS=8 GPU_MEM_UTIL=0.35 \
#     bash recipe/agentic/scripts/qwen36-27b-sft-datav3-no-think.sh
#
# 可视化（SwanLab 离线模式，无需联网）：
#   swanlab upload /mnt/workspace/agentic_rl/rl/runs/swanlab_logs
# ======================================================================
set -euo pipefail

# ---------- 根目录（所有文件都在这下面） ----------
BASE_DIR="${BASE_DIR:-/mnt/workspace/agentic_rl/rl/runs}"

# ---------- 每次运行的唯一 ID（可外部传入以复用同一目录） ----------
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# ---------- 路径与基础参数 ----------
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/agentic_rl/agentic_sft/sft_checkpoints/qwen3_6_sft_datav3_no_think}"
AI2C_REPO_PATH="${AI2C_REPO_PATH:-/mnt/workspace/agentic_rl/rl/ai2c}"
DATA_DIR="${DATA_DIR:-/mnt/workspace/agentic_rl/rl/data/ai_to_c_2w}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/hydra/grpo_train_${RUN_ID}}"
SESSION_DIR="${SESSION_DIR:-${BASE_DIR}/sessions}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/checkpoints/grpo_qwen36_27b_${RUN_ID}}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${BASE_DIR}/swanlab_logs/grpo_qwen36_27b_${RUN_ID}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-/mnt/workspace/agentic_rl/rl/rollout_logs/grpo_qwen36_27b_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"

# ---------- vLLM ----------
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3.6-27b-rl}"

# ---------- 规模参数 ----------
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"              # 27B rollout tensor parallel

# free_cache_engine=true 会在 log_prob 结束后释放 KV cache，训练阶段不占 vLLM 显存，
# 因此 rollout 阶段可以给 vLLM 更多比例（0.45），而不影响 backward 的可用显存
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"

SAVE_FREQ="${SAVE_FREQ:-50}"
KEEP_CKPTS="${KEEP_CKPTS:-10}"
EVAL_FREQ="${EVAL_FREQ:-50}"

# batch size —— 8 卡可行最低：train_batch_size=8（每卡 1 条 prompt）
# 约束：train_batch_size * rollout.n 必须能被 n_gpus 整除
#       rollout.n=4 时，train_batch_size 最小为 2；取 8 即每卡 1 prompt
BATCH_SIZE="${BATCH_SIZE:-8}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-8}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
LOG_PROB_MICRO="${LOG_PROB_MICRO:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"

# dynamic bsz token budget：
# 必须 >= 训练侧实际最长序列（MAX_PROMPT_LEN + MAX_RESP_LEN = 26000），加 buffer 取 28000
# 注意：这里故意与 MAX_MODEL_LEN(40960) 不同，因为训练侧会先截断再送入 backward
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-28000}"

# 【rollout 侧】外部 vLLM 的 context 上限
# 调大到 40960 防止 agentic 长对话（history 累积）触发 vLLM 400 error
# 注意：这个参数不影响 backward 显存，只控制 vLLM 允许的最大输入长度
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"

# 【训练侧截断】进入 backward 的序列长度上限
# prompt + resp 最长 = 14000 + 12000 = 26000，backward 不超过这个值
# rollout 生成的超长序列在这里被截断，与 MAX_MODEL_LEN 解耦
# prompt P99 ≈ 9k tokens（system+历史+当前query），加 buffer 取 14k
# resp  P99 ≈ 7k tokens（tool call+observation+最终回复），加 buffer 取 12k
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-9000}"
MAX_RESP_LEN="${MAX_RESP_LEN:-17000}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

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

LOG_FILE="${LOG_DIR}/grpo_train_${RUN_ID}.log"

echo "========================================"
echo " Agentic GRPO 正式训练 — Qwen3.6 27B"
echo "========================================"
echo " RUN_ID:       ${RUN_ID}"
echo " Model:        ${MODEL_PATH}"
echo " ai2c repo:    ${AI2C_REPO_PATH}"
echo " vLLM:         ${VLLM_BASE_URL}  (${VLLM_MODEL_NAME})"
echo " Data:         ${DATA_DIR}"
echo " Output:       ${OUTPUT_DIR}"
echo " Checkpoints:  ${CKPT_DIR}  (keep=${KEEP_CKPTS}，freq=每 ${SAVE_FREQ} step)"
echo " SwanLab:      ${SWANLAB_LOG_DIR}"
echo " Rollout logs: ${ROLLOUT_DATA_DIR}"
echo " Sessions:     ${SESSION_DIR}"
echo " Log:          ${LOG_FILE}"
echo " GPUs:         ${NUM_GPUS}  |  TP: ${TP_SIZE}"
echo " Steps:        ${TOTAL_STEPS}  |  Epochs: ${TOTAL_EPOCHS}"
echo " BatchSize:    ${BATCH_SIZE}  |  MiniBS: ${PPO_MINI_BATCH}  |  MicroBS: ${PPO_MICRO_BATCH}"
echo " Rollout.n:    ${ROLLOUT_N}"
echo " SeqLen:       prompt=${MAX_PROMPT_LEN}  resp=${MAX_RESP_LEN}  model=${MAX_MODEL_LEN}"
echo " GPU MemUtil:  ${GPU_MEM_UTIL}  |  MaxNumSeqs: ${MAX_NUM_SEQS}  |  FreeCacheEngine: ${FREE_CACHE_ENGINE}"
echo " FSDP:         fsdp1 + param/optimizer offload  |  entropy_coeff=0"
echo " DynBSZ:       ppo_max_token_len=${PPO_MAX_TOKEN_LEN}"
echo "========================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
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
    trainer.experiment_name=grpo_qwen36_27b_fsdp1_free_cache \
    +trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size=8 \
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
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
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
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/agentic/config/agent_loop_list.yaml \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    +agentic.apply_chat_template_kwargs.enable_thinking=false \
    +agentic.reward.normalize=false \
    +agentic.reward.clip_min=0.0 \
    +agentic.reward.clip_max=1.0 \
    "+agentic.reward.components=[{_target_: recipe.agentic.reward.components.AiToCEfficientReward, weight: 1.0, target_steps: 3, max_steps: 7}]" \
    \
    ray_kwargs.ray_init.num_cpus=32 \
    2>&1 | tee "${LOG_FILE}"

echo "========================================"
echo " Training finished."
echo " Log:         ${LOG_FILE}"
echo " Checkpoints: ${CKPT_DIR}"
echo " SwanLab:     swanlab upload ${SWANLAB_LOG_DIR}"
echo "========================================"
