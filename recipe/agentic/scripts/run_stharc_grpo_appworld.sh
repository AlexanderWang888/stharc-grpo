#!/usr/bin/env bash
# ======================================================================
# STHARC-GRPO 训练启动脚本 — AppWorld（真实数据 / 软依赖 credit assignment）
# ======================================================================
# 用法（在 training_code_public 目录下执行）：
#   cd ${WS}/research/training_code_public
#   export APPWORLD_ROOT=${WS}/research/appworld_explore   # 含 data/ 的父目录
#   # 禁止 pip install appworld（PyPI 0.1.3 会把 pydantic 降到 1.x，打坏 vLLM）
#   pip install "git+https://github.com/StonyBrookNLP/appworld.git" && appworld install
#   bash recipe/agentic/scripts/run_stharc_grpo_appworld.sh
#
# smoke / 摸管线（每卡 1 条；CTX 须 ≥ AppWorld system≈900tok，勿再用 512）：
#   # 旧紧预算（会 packing 全截）：MAX_RESP_LEN=1024 MAX_TASK_CONTEXT_LEN=512
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-2B \
#   NUM_GPUS=8 TP_SIZE=1 GPU_MEM_UTIL=0.40 \
#   BATCH_SIZE=1 ROLLOUT_N=8 NUM_AGENT_WORKERS=2 VAL_BATCH_SIZE=8 \
#   TOTAL_STEPS=2 SAVE_FREQ=0 TEST_FREQ=0 VAL_BEFORE_TRAIN=False \
#   K_MAX_CAP=10 MAX_PROMPT_LEN=4096 MAX_RESP_LEN=2048 MAX_TASK_CONTEXT_LEN=2048 \
#   MAX_MODEL_LEN=40960 APPWORLD_MAX_STEPS=30 USE_DYNAMIC_BSZ=false \
#   STHARC_RUN_FROZEN=false STHARC_LAM=0.25 STHARC_ATTRIBUTION=latest_only \
#   DATA_DIR=data/stharc_appworld_k6_10_smoke \
#   bash recipe/agentic/scripts/run_stharc_grpo_appworld.sh
#   # CHAIN_RESP = 10*2048+9*2048 = 38912；OOM 再降 MAX_RESP_LEN=1536
#   # 约束：BATCH×ROLLOUT_N 必须被 NUM_GPUS / NUM_AGENT_WORKERS 整除
#
# 数据集准备（先于训练运行）：
#   python3 ../dataset_explore/idea2_datasets/build_appworld_dataset.py \
#       --out-dir ../dataset_explore/idea2_datasets_out \
#       --parquet-dir data/stharc_appworld \
#       --G-train 4 --G-val 1
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 有些服务器只装了 python3，没有 python 软链；可用 PYTHON_BIN 覆盖
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

# 默认 main_ppo，避开 main_ppo_sync+TransferQueue 的等长 densify 崩溃
PPO_ENTRY="${PPO_ENTRY:-verl.trainer.main_ppo}"
if [[ "${PPO_ENTRY}" == *main_ppo_sync* && "${ALLOW_PPO_SYNC:-0}" != "1" ]]; then
  echo "[STHARC-appworld] ERROR: PPO_ENTRY=${PPO_ENTRY} 易 .offsets() 崩溃；改回 main_ppo（ALLOW_PPO_SYNC=1 可强制 sync）"
  PPO_ENTRY=verl.trainer.main_ppo
fi
echo "[STHARC-appworld] PPO entry: ${PPO_ENTRY}"

# ---------- APPWORLD_ROOT 检查 ----------
: "${APPWORLD_ROOT:=${VERL_RESEARCH_DIR}/../appworld_explore}"
export APPWORLD_ROOT
if [[ ! -d "${APPWORLD_ROOT}/data/tasks" ]]; then
    echo "[STHARC-appworld] WARNING: 未在 ${APPWORLD_ROOT}/data/tasks 找到本地 AppWorld 数据缓存"
    echo "[STHARC-appworld] 请确认 APPWORLD_ROOT 指向正确路径，或先补齐 data/ 缓存。"
fi
python3 -c "import appworld" 2>/dev/null || {
    echo "[STHARC-appworld] WARNING: 当前 Python 环境未安装 appworld 包，rollout 会在 AppWorldEnv.run() 里直接报错。"
    echo "[STHARC-appworld]   pip install \"git+https://github.com/StonyBrookNLP/appworld.git\" && appworld install"
    echo "[STHARC-appworld]   （勿用 PyPI appworld==0.1.3：会降级 pydantic 打坏 vLLM）"
}

# ---------- 全部落到大盘 /mnt/workspace ----------
STHARC_HOME="${STHARC_HOME:-/mnt/workspace/stharc}"
BASE_DIR="${BASE_DIR:-${STHARC_HOME}/runs/appworld}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-${VERL_RESEARCH_DIR}/data/stharc_appworld}"

export TMPDIR="${TMPDIR:-${STHARC_HOME}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${STHARC_HOME}/ray}"
export VERL_ZMQ_IPC_DIR="${VERL_ZMQ_IPC_DIR:-${STHARC_HOME}/zmq}"
export HF_HOME="${HF_HOME:-${STHARC_HOME}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export STHARC_SANDBOX_DIR="${STHARC_SANDBOX_DIR:-${STHARC_HOME}/rl_stharc}"

mkdir -p "${BASE_DIR}" "${TMPDIR}" "${RAY_TMPDIR}" "${VERL_ZMQ_IPC_DIR}" \
         "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${STHARC_SANDBOX_DIR}"
echo "[STHARC-appworld] STHARC_HOME=${STHARC_HOME}"
echo "[STHARC-appworld] BASE_DIR=${BASE_DIR}"
echo "[STHARC-appworld] TMPDIR/RAY/ZMQ=${TMPDIR} | ${RAY_TMPDIR} | ${VERL_ZMQ_IPC_DIR}"
echo "[STHARC-appworld] HF_HOME/SANDBOX=${HF_HOME} | ${STHARC_SANDBOX_DIR}"

# ---------- 路径与基础参数 ----------
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/Qwen3.5-0.8B}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/hydra/stharc_grpo_appworld_${RUN_ID}}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/checkpoints/stharc_grpo_appworld_${RUN_ID}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${BASE_DIR}/rollout_logs/stharc_grpo_appworld_${RUN_ID}}"
AGENT_LOOP_LIST_PATH="${AGENT_LOOP_LIST_PATH:-recipe/agentic/config/agent_loop_list.yaml}"

# ---------- vLLM ----------
# 默认让 verl 自己管理/探测内部 vLLM rollout 引擎地址（GlobalRequestLoadBalancer
# 探测 actor_rollout_ref.rollout.name=vllm 起的那个引擎，不需要你手动起一个独立
# 的外部 vLLM server）。VLLM_BASE_URL/VLLM_MODEL_NAME 仅作探测失败时的兜底值。
# 如果你想强制用手动起的外部 server（跳过自动探测），设
# STHARC_DISABLE_VLLM_AUTODETECT=true。
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${MODEL_PATH}}"
STHARC_DISABLE_VLLM_AUTODETECT="${STHARC_DISABLE_VLLM_AUTODETECT:-false}"

# ---------- 规模参数 ----------
NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"

BATCH_SIZE="${BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-4}"          # G = GRPO group size
# 与 public 脚本一致：gen batch（≈ BATCH×N）必须能被 num_workers 整除
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-$((BATCH_SIZE * ROLLOUT_N))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${NUM_AGENT_WORKERS}}"

# ── 长度预算（必须区分「单任务」与「整链」；对齐 public，否则 K≈19 会撑爆 nested tensor）──
# MAX_PROMPT_LEN          : task0 的 prompt 上限
# MAX_RESP_LEN            : **单任务** rollout 上限（AppWorld 默认更紧，因 K 更大）
# MAX_TASK_CONTEXT_LEN    : 拼链时后续 task 的 prompt 插入预算（mask=0）
# K_MAX / K_MAX_CAP       : 预算用的最大链长（默认扫 parquet；可覆盖）
# CHAIN_RESP_LEN          : 整链 response = K*RESP + (K-1)*CTX
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESP_LEN="${MAX_RESP_LEN:-2048}"
# 正式实验建议 = MAX_PROMPT_LEN，避免后续 task prompt packing 截断
MAX_TASK_CONTEXT_LEN="${MAX_TASK_CONTEXT_LEN:-${MAX_PROMPT_LEN}}"
K_MAX_CAP="${K_MAX_CAP:-8}"
STRICT_CHAIN_BUDGET="${STRICT_CHAIN_BUDGET:-0}"
ABORT_ON_PACKING_TRUNCATION="${ABORT_ON_PACKING_TRUNCATION:-false}"
ABORT_ON_EXEC_TRUNCATION="${ABORT_ON_EXEC_TRUNCATION:-false}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-30}"
TEST_FREQ="${TEST_FREQ:-30}"
# 设 VAL_BEFORE_TRAIN=False 可跳过开训前整表 val（smoke 摸管线常用）
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
PER_ROLLOUT_TIMEOUT="${PER_ROLLOUT_TIMEOUT:-900}"
# dynamic_bsz 在超长 AppWorld 链上更容易踩 nested-tensor unbind；smoke 可 USE_DYNAMIC_BSZ=false
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
# Qwen3.5 mRoPE：9B 在 remove-padding 路径上易 RoPE "seq vs 4"；信号测试可试 false（走 padded）
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"

# ---------- AppWorld env 专有参数 ----------
APPWORLD_MAX_STEPS="${APPWORLD_MAX_STEPS:-30}"
# reward：官方 world.evaluate()；门槛/信号体检可 APPWORLD_PARTIAL_CREDIT=true（主表用 false）
APPWORLD_PARTIAL_CREDIT="${APPWORLD_PARTIAL_CREDIT:-false}"
case "${APPWORLD_PARTIAL_CREDIT}" in
  1|true|TRUE|yes|YES|on|ON) APPWORLD_PARTIAL_CREDIT=true ;;
  *) APPWORLD_PARTIAL_CREDIT=false ;;
esac
# 1：给 AppWorldEnv 拼 format few-shot（强调 apis.* 必须经 execute，勿当工具名）
STHARC_FORMAT_FEW_SHOT="${STHARC_FORMAT_FEW_SHOT:-0}"
export STHARC_FORMAT_FEW_SHOT

# ---------- STHARC 专有参数（K 不固定；只有 GRPO group size G 可配置）───────────
G="${G:-4}"
STHARC_GAMMA="${STHARC_GAMMA:-0.9}"
STHARC_LAM="${STHARC_LAM:-0.25}"
STHARC_CREDIT="${STHARC_CREDIT:-pu}"
STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-latest_only}"
STHARC_SHUFFLE_SEED="${STHARC_SHUFFLE_SEED:-0}"
STHARC_RUN_FROZEN="${STHARC_RUN_FROZEN:-false}"
STHARC_BASE_USER_ID="${STHARC_BASE_USER_ID:-default}"

# ---------- 目录准备 ----------
mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}" "${ROLLOUT_DATA_DIR}" "${BASE_DIR}/logs"

# ---------- 数据路径验证 + 扫描 K_MAX ----------
if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "[STHARC-appworld] WARNING: train.parquet not found at ${DATA_DIR}/train.parquet"
    echo "[STHARC-appworld] 请先运行数据构造脚本："
    echo "  python3 ${VERL_RESEARCH_DIR}/../dataset_explore/idea2_datasets/build_appworld_dataset.py \\"
    echo "      --out-dir ${VERL_RESEARCH_DIR}/../dataset_explore/idea2_datasets_out \\"
    echo "      --parquet-dir ${DATA_DIR} --G-train ${G} --G-val 1"
fi

if [[ -z "${K_MAX:-}" ]]; then
    if [[ -f "${DATA_DIR}/train.parquet" ]]; then
        _K_SCAN="$("${PYTHON_BIN}" - <<PY
import json
import pyarrow.parquet as pq
df = pq.read_table("${DATA_DIR}/train.parquet", columns=["task_chain", "chain_g"]).to_pandas()
ks = []
for _, r in df.iterrows():
    if int(r.get("chain_g") or 0) != 0:
        continue
    tc = r["task_chain"]
    if isinstance(tc, str):
        tc = json.loads(tc)
    ks.append(len(tc) if isinstance(tc, list) else 1)
print(max(ks) if ks else 8)
PY
)"
        if (( _K_SCAN > K_MAX_CAP )); then
            if [[ "${STRICT_CHAIN_BUDGET}" == "1" || "${STRICT_CHAIN_BUDGET}" == "true" ]]; then
                echo "[STHARC-appworld] ERROR: parquet max K=${_K_SCAN} > K_MAX_CAP=${K_MAX_CAP}。"
                echo "[STHARC-appworld] STRICT_CHAIN_BUDGET=1：拒绝静默截断。请提高 K_MAX_CAP/K_MAX，或过滤长链，或 STRICT_CHAIN_BUDGET=0（不推荐）。"
                exit 1
            fi
            echo "[STHARC-appworld] WARNING: parquet max K=${_K_SCAN} > K_MAX_CAP=${K_MAX_CAP}；预算按 K_MAX=${K_MAX_CAP} 计（更长链 packing 截断，abort_on_packing 时会炸）。"
            K_MAX="${K_MAX_CAP}"
        else
            K_MAX="${_K_SCAN}"
        fi
    else
        K_MAX="${K_MAX_CAP}"
    fi
fi
K_MAX="${K_MAX}"

if [[ -z "${CHAIN_RESP_LEN:-}" ]]; then
    if (( K_MAX > 1 )); then
        CHAIN_RESP_LEN=$((K_MAX * MAX_RESP_LEN + (K_MAX - 1) * MAX_TASK_CONTEXT_LEN))
    else
        CHAIN_RESP_LEN="${MAX_RESP_LEN}"
    fi
fi
# PPO packing 仍按「prompt + 整链 response」预算。
# vLLM max_model_len 必须单独抬高：AppWorldEnv 走 HTTP 多轮 ReAct，历史可远超
# 单任务 packing（此前 12k 会在 ~11k prompt + 1536 max_tokens 处 400）。
# 不改 max_tokens / MAX_RESP_LEN；0.8B 显存足够开 32k 上下文。
_CHAIN_MODEL_LEN=$((MAX_PROMPT_LEN + CHAIN_RESP_LEN))
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-${_CHAIN_MODEL_LEN}}"
_DEFAULT_VLLM_LEN=32768
if (( _CHAIN_MODEL_LEN > _DEFAULT_VLLM_LEN )); then
    _DEFAULT_VLLM_LEN="${_CHAIN_MODEL_LEN}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${_DEFAULT_VLLM_LEN}}"
echo "[STHARC-appworld] budget: K_MAX=${K_MAX} CHAIN_RESP_LEN=${CHAIN_RESP_LEN} MAX_MODEL_LEN=${MAX_MODEL_LEN} STRICT=${STRICT_CHAIN_BUDGET} abort_pack=${ABORT_ON_PACKING_TRUNCATION}"

# Qwen3.5 mRoPE：每卡 gen>1 时 remove-padding pack 易把 (4,seq) 轴弄错 → RoPE "seq vs 4"
_GEN_BATCH=$((BATCH_SIZE * ROLLOUT_N))
if (( NUM_GPUS > 0 && _GEN_BATCH % NUM_GPUS == 0 )); then
    _GEN_PER_GPU=$((_GEN_BATCH / NUM_GPUS))
else
    _GEN_PER_GPU=-1
fi
if (( _GEN_PER_GPU > 1 )); then
    echo "[STHARC-appworld] WARNING: gen_per_gpu=${_GEN_PER_GPU} (BATCH×N/GPUS=${BATCH_SIZE}×${ROLLOUT_N}/${NUM_GPUS})>1；Qwen3.5 易在 compute_log_prob RoPE 炸。对齐 M4-lite：BATCH×N==NUM_GPUS 且 USE_DYNAMIC_BSZ=false。"
fi
if [[ "${USE_DYNAMIC_BSZ}" == "true" ]]; then
    echo "[STHARC-appworld] WARNING: USE_DYNAMIC_BSZ=true 在 AppWorld 长链上更易踩 mRoPE/nested；smoke 请 false。"
fi

echo "============================================================"
echo " STHARC-GRPO(appworld) 启动参数"
echo "  MODEL_PATH       : ${MODEL_PATH}"
echo "  APPWORLD_ROOT    : ${APPWORLD_ROOT}"
echo "  DATA_DIR         : ${DATA_DIR}"
echo "  CKPT_DIR         : ${CKPT_DIR}"
echo "  NUM_GPUS/TP_SIZE : ${NUM_GPUS} / ${TP_SIZE}"
echo "  BATCH/ROLLOUT_N  : ${BATCH_SIZE} / ${ROLLOUT_N}"
echo "  NUM_AGENT_WORKERS: ${NUM_AGENT_WORKERS}  (must divide BATCH×N)"
echo "  VAL_BATCH_SIZE   : ${VAL_BATCH_SIZE}"
echo "  G / K_MAX        : ${G} / ${K_MAX}  (实际 K 由每条 chain 决定)"
echo "  MAX_PROMPT_LEN   : ${MAX_PROMPT_LEN}"
echo "  MAX_RESP_LEN     : ${MAX_RESP_LEN}  (单任务)"
echo "  MAX_TASK_CTX_LEN : ${MAX_TASK_CONTEXT_LEN}"
echo "  CHAIN_RESP_LEN   : ${CHAIN_RESP_LEN}"
echo "  MAX_MODEL_LEN    : ${MAX_MODEL_LEN}"
echo "  PPO_MAX_TOKEN_LEN: ${PPO_MAX_TOKEN_LEN}"
echo "  USE_DYNAMIC_BSZ  : ${USE_DYNAMIC_BSZ}"
echo "  USE_REMOVE_PADDING: ${USE_REMOVE_PADDING}"
echo "  APPWORLD_MAX_STEPS: ${APPWORLD_MAX_STEPS}"
echo "  REWARD           : AppWorldVerifierReward (official evaluate success 0/1)"
echo "  FORMAT_FEW_SHOT  : ${STHARC_FORMAT_FEW_SHOT}"
echo "  GAMMA / LAM      : ${STHARC_GAMMA} / ${STHARC_LAM}"
echo "  ATTRIBUTION      : ${STHARC_ATTRIBUTION}"
echo "  RUN_FROZEN       : ${STHARC_RUN_FROZEN}"
echo "  PARTIAL_CREDIT  : ${APPWORLD_PARTIAL_CREDIT}  (gate/smoke=true; main=false)"
echo "  TOTAL_STEPS      : ${TOTAL_STEPS}"
echo "============================================================"

# ---------- 启动训练 ----------
# 注意：不再用 --config-path/--config-name 指向自建的独立 stharc_grpo_appworld.yaml。
# 原因同 run_stharc_grpo_public.sh 头部注释：那份 yaml 用 `defaults: - _self_`，
# 完全绕开了 verl 官方 ppo_trainer.yaml 的 defaults 组合树（critic/reward/model/
# rollout 等子 schema 不会被组合进来），导致 need_critic()/need_reward_model()
# 访问 config.critic.enable / config.reward.reward_model.* 时报
# "Key 'enable' is not in struct"。改为不传 --config-path/--config-name，让
# hydra 走 main_ppo.py 默认 config_name="ppo_trainer"；STHARC 用 +stharc.xxx 追加
"${PYTHON_BIN}" -m "${PPO_ENTRY}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode=disable \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=stharc_train_appworld \
    trainer.experiment_name=stharc_grpo_appworld \
    trainer.default_local_dir="${CKPT_DIR}" \
    +trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    critic.enable=false \
    reward.reward_model.enable=false \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${CHAIN_RESP_LEN}" \
    data.trust_remote_code=True \
    data.return_raw_chat=true \
    data.shuffle=true \
    data.filter_overlong_prompts=false \
    +data.data_source=stharc \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_LIST_PATH}" \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.response_length="${CHAIN_RESP_LEN}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    +stharc.env._target_=recipe.agentic.envs.AppWorldEnv \
    "+stharc.env.max_steps=${APPWORLD_MAX_STEPS}" \
    "+stharc.env.appworld_root=${APPWORLD_ROOT}" \
    "+stharc.env.format_few_shot=${STHARC_FORMAT_FEW_SHOT}" \
    +stharc.harness_state_module=recipe.agentic.stharc.generic_harness_state \
    "+stharc.vllm_base_url=${VLLM_BASE_URL}" \
    "+stharc.vllm_model_name=${VLLM_MODEL_NAME}" \
    "+stharc.disable_vllm_autodetect=${STHARC_DISABLE_VLLM_AUTODETECT}" \
    "+stharc.per_rollout_timeout_sec=${PER_ROLLOUT_TIMEOUT}" \
    "+stharc.gamma=${STHARC_GAMMA}" \
    "+stharc.credit=${STHARC_CREDIT}" \
    "+stharc.lam=${STHARC_LAM}" \
    "+stharc.attribution=${STHARC_ATTRIBUTION}" \
    "+stharc.shuffle_seed=${STHARC_SHUFFLE_SEED}" \
    "+stharc.run_frozen_chain=${STHARC_RUN_FROZEN}" \
    "+stharc.base_user_id=${STHARC_BASE_USER_ID}" \
    "+stharc.max_task_response_length=${MAX_RESP_LEN}" \
    "+stharc.max_task_context_length=${MAX_TASK_CONTEXT_LEN}" \
    "+stharc.abort_on_packing_truncation=${ABORT_ON_PACKING_TRUNCATION}" \
    "+stharc.abort_on_exec_truncation=${ABORT_ON_EXEC_TRUNCATION}" \
    "+stharc.reward.components=[{_target_: recipe.agentic.reward.AppWorldVerifierReward, weight: 1.0, enabled: true, partial_credit: ${APPWORLD_PARTIAL_CREDIT}}]" \
    +stharc.reward.normalize=false \
    +stharc.reward.clip_min=0.0 \
    +stharc.reward.clip_max=1.0 \
    \
    ray_kwargs.ray_init.num_cpus=32 \
    2>&1 | tee "${BASE_DIR}/logs/stharc_grpo_appworld_${RUN_ID}.log"
