#!/usr/bin/env bash
# ======================================================================
# STHARC-GRPO 训练启动脚本 — 公开数据集 / 业务解耦版（idea2 验证用）
# ======================================================================
# 与 run_stharc_grpo.sh 的关键差异：不依赖 ai2c 业务代码/_research_workspace_paths.sh
# （该共享脚本默认配置 AI2C_REPO_PATH 等业务路径，可由环境变量覆盖）；env 换成
# recipe.agentic.envs.GenericBenchmarkEnv，数据换成 6 个公开/合成数据集混合
# parquet（K 由内容参数决定，不是固定 3）。
#
# 用法（在 training_code_public 目录下执行）：
#   cd ${WS}/research/training_code_public
#   bash recipe/agentic/scripts/run_stharc_grpo_public.sh
#
# smoke / 三角短训（2卡）。注意：MAX_RESP_LEN 是「单任务」预算，不是整链！
# 整链 CHAIN_RESP_LEN 会按 K_MAX 自动算；也可显式覆盖。
#   NUM_GPUS=2 TP_SIZE=1 BATCH_SIZE=4 ROLLOUT_N=2 TOTAL_STEPS=20 SAVE_FREQ=0 \
#   MAX_PROMPT_LEN=4096 MAX_RESP_LEN=2048 MAX_TASK_CONTEXT_LEN=512 \
#   GPU_MEM_UTIL=0.5 DATA_DIR=data/stharc_public_smoke MODEL_PATH=... \
#   bash recipe/agentic/scripts/run_stharc_grpo_public.sh
#
# 测评-only：
#   EVAL_ONLY=1 NUM_GPUS=2 ... bash recipe/agentic/scripts/run_stharc_grpo_public.sh
#   TRACE_DIR=${ROLLOUT_DATA_DIR}/traces bash recipe/agentic/scripts/run_credit_align_eval.sh
#
# 数据集准备（K 由内容决定，不需要 --K）：
#   python3 ../dataset_explore/idea2_datasets/build_idea2_datasets.py \
#       --out-dir ../dataset_explore/idea2_datasets_out \
#       --parquet-dir data/stharc_public \
#       --n-per-dataset 300 --val-ratio 0.1 --G-train 4 --G-val 1
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 有些服务器只装了 python3，没有 python 软链；可用 PYTHON_BIN 覆盖
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

# 默认走 main_ppo（与 qwen08b 生产脚本一致），避开 main_ppo_sync+TransferQueue
# 等长 response_mask densify 后 .offsets() 崩溃。
# 需要 sync 时显式：ALLOW_PPO_SYNC=1 PPO_ENTRY=verl.trainer.main_ppo_sync
PPO_ENTRY="${PPO_ENTRY:-verl.trainer.main_ppo}"
if [[ "${PPO_ENTRY}" == *main_ppo_sync* && "${ALLOW_PPO_SYNC:-0}" != "1" ]]; then
  echo "[STHARC-public] ERROR: PPO_ENTRY=${PPO_ENTRY} 会触发 response_mask.offsets() 崩溃。"
  echo "[STHARC-public]        已自动改回 verl.trainer.main_ppo；确需 sync 请加 ALLOW_PPO_SYNC=1"
  PPO_ENTRY=verl.trainer.main_ppo
fi
echo "[STHARC-public] PPO entry: ${PPO_ENTRY}"

# ---------- 全部落到大盘 /mnt/workspace（短路径，兼顾 AF_UNIX ≤107）----------
# 覆盖：训练产物、Ray、ZMQ、HF 缓存、通用 TMP、STHARC sandbox。
STHARC_HOME="${STHARC_HOME:-/mnt/workspace/stharc}"
BASE_DIR="${BASE_DIR:-${STHARC_HOME}/runs/public}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-${VERL_RESEARCH_DIR}/data/stharc_public}"

export TMPDIR="${TMPDIR:-${STHARC_HOME}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${STHARC_HOME}/ray}"
export VERL_ZMQ_IPC_DIR="${VERL_ZMQ_IPC_DIR:-${STHARC_HOME}/zmq}"
export HF_HOME="${HF_HOME:-${STHARC_HOME}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export STHARC_SANDBOX_DIR="${STHARC_SANDBOX_DIR:-${STHARC_HOME}/rl_stharc}"

# ---------- 路径与基础参数 ----------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/hydra/stharc_grpo_public_${RUN_ID}}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/checkpoints/stharc_grpo_public_${RUN_ID}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${BASE_DIR}/rollout_logs/stharc_grpo_public_${RUN_ID}}"
AGENT_LOOP_LIST_PATH="${AGENT_LOOP_LIST_PATH:-recipe/agentic/config/agent_loop_list.yaml}"

mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${VERL_ZMQ_IPC_DIR}" \
         "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${STHARC_SANDBOX_DIR}"

# ---------- vLLM ----------
# 默认让 verl 自己管理/探测内部 vLLM rollout 引擎地址（GlobalRequestLoadBalancer
# 探测 actor_rollout_ref.rollout.name=vllm 起的那个引擎，不需要你手动起一个独立
# 的外部 vLLM server）。VLLM_BASE_URL/VLLM_MODEL_NAME 仅作探测失败时的兜底值。
# 如果你想强制用手动起的外部 server（跳过自动探测），设
# STHARC_DISABLE_VLLM_AUTODETECT=true。
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen-7b-rl}"
STHARC_DISABLE_VLLM_AUTODETECT="${STHARC_DISABLE_VLLM_AUTODETECT:-false}"

# ---------- 规模参数 ----------
NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"

BATCH_SIZE="${BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"          # G = GRPO group size
# AgentLoop 要求 gen batch（≈ BATCH×N）能被 num_workers 整除；默认对齐到 BATCH×N，
# 避免小 smoke（如 2×2=4）撞上 rollout.yaml 默认 num_workers=8。
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-$((BATCH_SIZE * ROLLOUT_N))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${NUM_AGENT_WORKERS}}"

# ── 长度预算（必须区分「单任务」与「整链」）────────────────────────────
# MAX_PROMPT_LEN          : task0 的 prompt 上限（system+tools+query）
# MAX_RESP_LEN            : **单任务** rollout 上限（Thought/Action/obs）
# MAX_TASK_CONTEXT_LEN    : 拼链时后续 task 的 prompt 插入预算（mask=0）
# K_MAX                   : 预算用的最大链长（默认从 parquet 扫描；可覆盖）
# CHAIN_RESP_LEN          : 整链 response 上限 = K*RESP + (K-1)*CTX
# MAX_MODEL_LEN / PPO_*   : 默认跟 prompt+chain 对齐，避免再被二次截断
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
# 论文正式单任务 response / 后续任务 context 预算；整链另按 K 计算
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_TASK_CONTEXT_LEN="${MAX_TASK_CONTEXT_LEN:-4096}"
# 预算用的 K：默认扫 parquet 真实 maxK。K_MAX_CAP 仅作上限（默认极大=不压预算）
# 注意：K_MAX_CAP 不会少跑 task，但会缩小 CHAIN_RESP_LEN → 长链 packing 截断（可被检测）
K_MAX_CAP="${K_MAX_CAP:-999}"
STRICT_CHAIN_BUDGET="${STRICT_CHAIN_BUDGET:-0}"
# 截断默认只检测/打点；需要时再开 abort
ABORT_ON_PACKING_TRUNCATION="${ABORT_ON_PACKING_TRUNCATION:-false}"
ABORT_ON_EXEC_TRUNCATION="${ABORT_ON_EXEC_TRUNCATION:-false}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"
PER_ROLLOUT_TIMEOUT="${PER_ROLLOUT_TIMEOUT:-600}"
# resume: disable | auto（找 CKPT_DIR 下最新 global_step_*）| resume_path（配合 RESUME_FROM_PATH）
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

# EVAL_ONLY=1：val_before_train；默认再开 VAL_ONLY=1，评完就停（不 dummy 训 1 步）。
# 若仍要那 1 步 train（旧 CreditAlign 用法）：EVAL_ONLY=1 VAL_ONLY=0
# 注意：verl 在最后一步仍会 save（只要 save_freq>0）；测评可不存大 ckpt，SAVE_FREQ=0。
EVAL_ONLY="${EVAL_ONLY:-0}"
if [[ "${EVAL_ONLY}" == "1" || "${EVAL_ONLY}" == "true" ]]; then
    TOTAL_STEPS="${TOTAL_STEPS_EVAL:-1}"
    SAVE_FREQ=0
    TEST_FREQ=1
    # 未显式 VAL_ONLY=0 时，评完就停（不 dummy 训 1 步）
    VAL_ONLY="${VAL_ONLY:-1}"
    echo "[STHARC-public] EVAL_ONLY=1 → TOTAL_STEPS=${TOTAL_STEPS}, SAVE_FREQ=0（不存 ckpt）, 重点跑 val + traces"
else
    VAL_ONLY="${VAL_ONLY:-0}"
fi
if [[ "${VAL_ONLY}" == "1" || "${VAL_ONLY}" == "true" ]]; then
    VAL_ONLY_HYDRA=True
    echo "[STHARC-public] VAL_ONLY=1 → trainer.val_only=True（评完退出，不更新权重）"
else
    VAL_ONLY_HYDRA=False
fi

# ---------- STHARC 专有参数（K 不固定；只有 GRPO group size G 可配置）───────────
G="${G:-4}"
STHARC_GAMMA="${STHARC_GAMMA:-0.9}"
STHARC_LAM="${STHARC_LAM:-0.25}"
STHARC_CREDIT="${STHARC_CREDIT:-pu}"
STHARC_ATTRIBUTION="${STHARC_ATTRIBUTION:-latest_only}"
STHARC_SHUFFLE_SEED="${STHARC_SHUFFLE_SEED:-0}"
# 论文多 seed：系统 RNG（DataLoader + actor FSDP + vLLM engine）。
# 不改 parquet / val。SEED 与 TRAIN_SEED 同义。例：SEED=123
SEED="${SEED:-${TRAIN_SEED:-}}"
TRAIN_SEED="${SEED}"
if [[ -n "${SEED}" ]]; then
  export PYTHONHASHSEED="${SEED}"
  STHARC_SHUFFLE_SEED="${SEED}"
fi
STHARC_RUN_FROZEN="${STHARC_RUN_FROZEN:-false}"
STHARC_BASE_USER_ID="${STHARC_BASE_USER_ID:-default}"
STHARC_MAX_STEPS="${STHARC_MAX_STEPS:-20}"
# 格式 few-shot（无 SFT 冷启动时建议 1）：只示范 Thought/Action，不背 diagnostic 题
STHARC_FORMAT_FEW_SHOT="${STHARC_FORMAT_FEW_SHOT:-0}"
export STHARC_FORMAT_FEW_SHOT
# actor 优化（2B 诊断易 KL 爆炸时可加大 KL / 略降 lr）
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
# 所有任务保留上游奖励，不按读写类型屏蔽或整形；ExactMatch 设置不变。
# 脏链（trunc / no_final / env fail）不上 PU → chain_reward=total_reward（方案 A）
STHARC_PU_REQUIRE_CLEAN="${STHARC_PU_REQUIRE_CLEAN:-false}"

# ---------- 目录准备 ----------
mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}" "${ROLLOUT_DATA_DIR}" "${BASE_DIR}/logs"

# ---------- 数据路径验证 + 扫描 K_MAX ----------
if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "[STHARC-public] WARNING: train.parquet not found at ${DATA_DIR}/train.parquet"
    echo "[STHARC-public] 请先运行数据构造脚本："
    echo "  python3 ${VERL_RESEARCH_DIR}/../dataset_explore/idea2_datasets/build_idea2_datasets.py \\"
    echo "      --out-dir ${VERL_RESEARCH_DIR}/../dataset_explore/idea2_datasets_out \\"
    echo "      --parquet-dir ${DATA_DIR} \\"
    echo "      --n-per-dataset 300 --val-ratio 0.1 --G-train ${G} --G-val 1"
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
                echo "[STHARC-public] ERROR: parquet max K=${_K_SCAN} > K_MAX_CAP=${K_MAX_CAP}。"
                echo "[STHARC-public] STRICT_CHAIN_BUDGET=1：拒绝静默截断。请提高 K_MAX_CAP/K_MAX，或过滤长链数据，或显式 STRICT_CHAIN_BUDGET=0（不推荐）。"
                exit 1
            fi
            echo "[STHARC-public] WARNING: parquet max K=${_K_SCAN} > K_MAX_CAP=${K_MAX_CAP}；预算按 K_MAX=${K_MAX_CAP} 计（更长链会 packing 截断，abort_on_packing 时会炸）。"
            K_MAX="${K_MAX_CAP}"
        else
            K_MAX="${_K_SCAN}"
        fi
    else
        K_MAX="${K_MAX_CAP}"
    fi
fi
K_MAX="${K_MAX}"

# 整链 response 预算（与 agent_loop._concat_builder_outputs 对齐）
if [[ -z "${CHAIN_RESP_LEN:-}" ]]; then
    if (( K_MAX > 1 )); then
        CHAIN_RESP_LEN=$((K_MAX * MAX_RESP_LEN + (K_MAX - 1) * MAX_TASK_CONTEXT_LEN))
    else
        CHAIN_RESP_LEN="${MAX_RESP_LEN}"
    fi
fi
# packing / PPO 预算：盖住 prompt0 + 整链 response（可远大于单任务 ReAct 窗口）
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LEN + CHAIN_RESP_LEN))}"
# vLLM 引擎窗口：单任务多轮 ReAct 用；不要跟 CHAIN_RESP 绑死（否则常被模型 max_pos 压回 16k）
# 显式 MAX_MODEL_LEN 仍可覆盖；默认 32768（需模型/rope 支持；不够再加）
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-32768}}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
echo "[STHARC-public] budget: K_MAX=${K_MAX} CHAIN_RESP_LEN=${CHAIN_RESP_LEN} VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} PPO_MAX=${PPO_MAX_TOKEN_LEN} STRICT=${STRICT_CHAIN_BUDGET} abort_pack=${ABORT_ON_PACKING_TRUNCATION}"

echo "============================================================"
echo " STHARC-GRPO(public) 启动参数"
echo "  MODEL_PATH       : ${MODEL_PATH}"
echo "  STHARC_HOME      : ${STHARC_HOME}"
echo "  BASE_DIR         : ${BASE_DIR}"
echo "  TMPDIR/RAY/ZMQ   : ${TMPDIR} | ${RAY_TMPDIR} | ${VERL_ZMQ_IPC_DIR}"
echo "  HF_HOME/SANDBOX  : ${HF_HOME} | ${STHARC_SANDBOX_DIR}"
echo "  DATA_DIR         : ${DATA_DIR}"
echo "  CKPT_DIR         : ${CKPT_DIR}"
echo "  ROLLOUT_DATA_DIR : ${ROLLOUT_DATA_DIR}"
echo "  NUM_GPUS/TP_SIZE : ${NUM_GPUS} / ${TP_SIZE}"
echo "  BATCH/ROLLOUT_N  : ${BATCH_SIZE} / ${ROLLOUT_N}"
echo "  NUM_AGENT_WORKERS: ${NUM_AGENT_WORKERS}  (must divide BATCH×N)"
echo "  VAL_BATCH_SIZE   : ${VAL_BATCH_SIZE}"
echo "  G / K_MAX        : ${G} / ${K_MAX}  (实际 K 由每条 chain 决定)"
echo "  MAX_PROMPT_LEN   : ${MAX_PROMPT_LEN}  (task0 prompt)"
echo "  MAX_RESP_LEN     : ${MAX_RESP_LEN}  (单任务 rollout)"
echo "  MAX_TASK_CTX_LEN : ${MAX_TASK_CONTEXT_LEN}  (后续 task prompt 插入)"
echo "  CHAIN_RESP_LEN   : ${CHAIN_RESP_LEN}  (= K*RESP + (K-1)*CTX)"
echo "  MAX_MODEL_LEN    : ${MAX_MODEL_LEN}"
echo "  PPO_MAX_TOKEN_LEN: ${PPO_MAX_TOKEN_LEN}"
echo "  GAMMA / LAM      : ${STHARC_GAMMA} / ${STHARC_LAM}"
echo "  CREDIT           : ${STHARC_CREDIT}  γ_curate=${STHARC_SKILLRISE_GAMMA:-0.6} curate_max=${STHARC_SKILLRISE_CURATE_MAX_TOKENS:-1024}"
echo "  SAGE_BONUS_COEF  : ${STHARC_SAGE_BONUS_COEF:-1.0}  require_self_success=${STHARC_SAGE_REQUIRE_SELF_SUCCESS:-true}"
echo "  ATTRIBUTION      : ${STHARC_ATTRIBUTION}"
echo "  SHUFFLE_SEED     : ${STHARC_SHUFFLE_SEED}"
echo "  SEED             : ${SEED:-null}  (data+fsdp+vLLM; parquet unchanged)"
echo "  RUN_FROZEN       : ${STHARC_RUN_FROZEN}"
echo "  FORMAT_FEW_SHOT  : ${STHARC_FORMAT_FEW_SHOT}"
echo "  ACTOR_LR / KL    : ${ACTOR_LR} / ${KL_LOSS_COEF}"
echo "  PU_REQUIRE_CLEAN : ${STHARC_PU_REQUIRE_CLEAN}"
echo "  SAVE_FREQ        : ${SAVE_FREQ}"
echo "  TEST_FREQ        : ${TEST_FREQ}"
echo "  TOTAL_STEPS      : ${TOTAL_STEPS}"
echo "  RESUME_MODE      : ${RESUME_MODE}${RESUME_FROM_PATH:+  from=${RESUME_FROM_PATH}}"
echo "  CKPT_DIR         : ${CKPT_DIR}"
echo "============================================================"

if [[ "${RESUME_MODE}" == "resume_path" && -z "${RESUME_FROM_PATH}" ]]; then
    echo "[STHARC-public] ERROR: RESUME_MODE=resume_path 但未设置 RESUME_FROM_PATH" >&2
    exit 2
fi

# ---------- 启动训练 ----------
# 注意：不再用 --config-path/--config-name 指向自建的独立 stharc_grpo_public.yaml
# （那份 yaml 用 `defaults: - _self_`，完全绕开了 verl 官方 ppo_trainer.yaml 的
#  defaults 组合树 —— critic/reward/model/rollout 等子 schema 都不会被组合进来，
#  导致 verl.trainer.ppo.utils.need_critic() 访问 config.critic.enable 时报
#  "Key 'enable' is not in struct"；同理 config.reward.reward_model.* 也会缺失）。
# 改为不传 --config-path/--config-name，让 hydra 走 main_ppo.py 的默认
# config_name="ppo_trainer"（会正确组合 critic/reward/model/rollout 等官方子
# schema，且都带安全默认值），STHARC 专属字段全部用 +stharc.xxx 追加
# （与 run_ai_to_c_32b_*.sh / run_stharc_qwen08b_*.sh 一致）。
"${PYTHON_BIN}" -m "${PPO_ENTRY}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    ${RESUME_FROM_PATH:+trainer.resume_from_path="${RESUME_FROM_PATH}"} \
    trainer.val_before_train=True \
    trainer.val_only=${VAL_ONLY_HYDRA} \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=stharc_train_public \
    trainer.experiment_name=stharc_grpo_public \
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
    ${SEED:+data.seed="${SEED}"} \
    ${SEED:+actor_rollout_ref.actor.data_loader_seed="${SEED}"} \
    ${SEED:+actor_rollout_ref.actor.fsdp_config.seed="${SEED}"} \
    ${SEED:+'+actor_rollout_ref.rollout.engine_kwargs.vllm.seed='"${SEED}"} \
    data.filter_overlong_prompts=false \
    +data.data_source=stharc \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
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
    actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    +stharc.env._target_=recipe.agentic.envs.GenericBenchmarkEnv \
    "+stharc.env.max_steps=${STHARC_MAX_STEPS}" \
    "+stharc.env.format_few_shot=${STHARC_FORMAT_FEW_SHOT}" \
    +stharc.harness_state_module=recipe.agentic.stharc.generic_harness_state \
    "+stharc.vllm_base_url=${VLLM_BASE_URL}" \
    "+stharc.vllm_model_name=${VLLM_MODEL_NAME}" \
    "+stharc.disable_vllm_autodetect=${STHARC_DISABLE_VLLM_AUTODETECT}" \
    "+stharc.per_rollout_timeout_sec=${PER_ROLLOUT_TIMEOUT}" \
    "+stharc.gamma=${STHARC_GAMMA}" \
    "+stharc.lam=${STHARC_LAM}" \
    "+stharc.credit=${STHARC_CREDIT}" \
    "+stharc.attribution=${STHARC_ATTRIBUTION}" \
    "+stharc.shuffle_seed=${STHARC_SHUFFLE_SEED}" \
    "+stharc.run_frozen_chain=${STHARC_RUN_FROZEN}" \
    "+stharc.base_user_id=${STHARC_BASE_USER_ID}" \
    "+stharc.max_task_response_length=${MAX_RESP_LEN}" \
    "+stharc.max_task_context_length=${MAX_TASK_CONTEXT_LEN}" \
    "+stharc.abort_on_packing_truncation=${ABORT_ON_PACKING_TRUNCATION}" \
    "+stharc.abort_on_exec_truncation=${ABORT_ON_EXEC_TRUNCATION}" \
    "+stharc.pu_require_clean=${STHARC_PU_REQUIRE_CLEAN}" \
    +stharc.reward.components="[{_target_: recipe.agentic.reward.ExactMatchReward, weight: 1.0, enabled: true, case_sensitive: false, strip_punctuation: true, numeric_tolerance: 1e-3, allow_substring: true, substring_slack_chars: 20}]" \
    +stharc.reward.normalize=false \
    +stharc.reward.clip_min=0.0 \
    +stharc.reward.clip_max=1.0 \
    \
    ray_kwargs.ray_init.num_cpus=32 \
    2>&1 | tee "${BASE_DIR}/logs/stharc_grpo_public_${RUN_ID}.log"
