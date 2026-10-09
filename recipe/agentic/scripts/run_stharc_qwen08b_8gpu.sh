#!/usr/bin/env bash
# ======================================================================
# STHARC-GRPO — Qwen3.5-0.8B  |  8 × GPU（默认 0-7，可通过 CUDA_VISIBLE_DEVICES 覆盖）
# 使用 main_ppo（与生产脚本一致），无需 TransferQueue
# K=3, G=4；STHARC_* 默认见 _research_workspace_paths.sh（含 RUN_FROZEN）
#
# Reward：对齐生产 pairwise（AiToCCombinedReward / GSB 正反双跑）
#   - 需配置 ANTHROPIC_API_KEY_POOL（及可选 ANTHROPIC_BASE_URL / ANTHROPIC_MODEL）
#   - 造数时每个 task 需带 baseline_answer（query 或 extra_info 字段会透传到 task_chain）
#   - 缺 baseline → reward=NaN → GRPO mask
# ======================================================================
# 用法（在 training_code_public 目录下执行）：
#   cd ${WS}/research/training_code_public
#   bash recipe/agentic/scripts/run_stharc_qwen08b_8gpu.sh
#
# 指定其它卡号（例如只用后 4 张）：
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 \
#   bash recipe/agentic/scripts/run_stharc_qwen08b_8gpu.sh
#
# smoke test（5 步验流程）：
#   TOTAL_STEPS=5 BATCH_SIZE=8 GPU_MEM_UTIL=0.65 \
#   bash recipe/agentic/scripts/run_stharc_qwen08b_8gpu.sh
#
# 换 K 做实验（数据与脚本必须同一 K；启动时会校验 parquet）：
#   # 1) 造数
#   python3 -m recipe.agentic.data.prepare_dataset stharc \
#       --train-input /path/to/train.jsonl \
#       --train-output ../data/stharc_2w/train.parquet \
#       --val-output ../data/stharc_2w/val.parquet \
#       --val-ratio 0.05 --K 5 --G 1 --strategy random --seed 42
#   # 2) 开训
#   K=5 bash recipe/agentic/scripts/run_stharc_qwen08b_8gpu.sh
# ======================================================================
set -euo pipefail

# ── GPU 绑定（默认 8 卡，与文件名一致）────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ── 路径（ai2c_workspace：共用 ${WS}/ai2c，数据在 research/data）────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_research_workspace_paths.sh"

SERVER_ROOT="${SERVER_ROOT}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/Qwen3.5-0.8B}"
# AI2C_REPO_PATH / DATA_DIR 已由 _research_workspace_paths.sh 设定（可 env 覆盖）

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SERVER_ROOT}/runs/hydra/stharc_${RUN_ID}}"
SESSION_DIR="${SESSION_DIR:-${SERVER_ROOT}/runs/sessions}"
CKPT_DIR="${CKPT_DIR:-${SERVER_ROOT}/runs/checkpoints/stharc_${RUN_ID}}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${SERVER_ROOT}/runs/swanlab/stharc_${RUN_ID}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${SERVER_ROOT}/runs/rollout_logs/stharc_${RUN_ID}}"
GSB_JUDGE_IO_LOG_DIR="${GSB_JUDGE_IO_LOG_DIR:-${ROLLOUT_DATA_DIR}/gsb_pairwise_io}"
ACCURACY_JUDGE_IO_LOG_DIR="${ACCURACY_JUDGE_IO_LOG_DIR:-${ROLLOUT_DATA_DIR}/accuracy_io}"
LOG_DIR="${LOG_DIR:-${SERVER_ROOT}/runs/logs}"

# 相对本脚本定位，不依赖 cwd
AGENT_LOOP_LIST_PATH="${AGENT_LOOP_LIST_PATH:-$(cd "${SCRIPT_DIR}/../config" && pwd)/agent_loop_list.yaml}"

# ── vLLM ──────────────────────────────────────────────────────────────
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
# VERL 内部 vLLM 的默认 model name = 完整 MODEL_PATH（prometheus 未开启时不设 served_model_name）
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${MODEL_PATH}}"

# ── 规模参数 ──────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE=1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"

# 【训练侧截断】进入 FSDP backward 的长度上限（与 vLLM MAX_MODEL_LEN 解耦）
# prompt = task0 的 system+tools+query；必须盖住完整 system（~12K+）+ tools + user，
# 截断是 [:N] 保开头砍尾部——过小会直接砍掉 user query。对齐生产 20480。
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-20480}"
# 单任务 rollout 上限（含 tool 观察）；agent_loop 会按任务强制截断，再拼 chain。
# 默认 8192：配合 REACT_MAX_TOKENS=1024 足够；过大只会抬高 pad / OOM。
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
# vLLM 上下文窗口：覆盖 ReAct 多轮累积（system+tools ~12K+ + 多轮对话）
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# 单次调度 token 预算：须 ≥ 单条最长 prompt（默认跟 MAX_MODEL_LEN），
# 否则走 verl 默认 8192，长 ReAct 上下文会被卡住/极慢。
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

BATCH_SIZE="${BATCH_SIZE:-32}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-${BATCH_SIZE}}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-2}"
# G：GRPO group size（每条 chain 并行采样 G 次）。
# STHARC 数据准备时 --G 1（每条 chain 在 parquet 中只存 1 行），
# trainer 通过 rollout.n=G 并行采样 G 次，GRPO 在同一 uid 的 G 个 rollout 之间计算 advantage。
# ROLLOUT_N 默认跟随 G，若需独立控制可显式设置 ROLLOUT_N。
G="${G:-4}"
ROLLOUT_N="${ROLLOUT_N:-${G}}"
# agent.num_workers：每个 worker 处理 1 条 gen sample；训练侧 gen 后为 BATCH*N。
# val 会 pad 到 num_workers 的倍数，故 VAL_BATCH_SIZE 默认与之对齐，避免 8→128 空跑。
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-$((BATCH_SIZE * ROLLOUT_N))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${NUM_AGENT_WORKERS}}"

TOTAL_STEPS="${TOTAL_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
PER_ROLLOUT_TIMEOUT="${PER_ROLLOUT_TIMEOUT:-300}"

# ── STHARC 参数 ───────────────────────────────────────────────────────
# K：chain 长度。必须与 parquet 造数时的 --K 一致；启动时会校验 train/val。
# 换 K 做实验时：先 prepare_dataset --K <N>，再用同一 K=<N> 开训。
K="${K:-3}"
# G 已在「规模参数」区定义（与 ROLLOUT_N 共同设置），此处不重复赋值

# CHAIN_RESP_LEN：送入训练框架的 response tensor 最大长度（也会 pad 到此长度）。
# 拼接结构（见 _concat_builder_outputs）：
#   response = r0 + (p1 + r1) + (p2 + r2) + ...
#   pk（k≥1）= 后续任务整段 prompt（mask=0），按 MAX_PROMPT_LEN 截断。
# 公式：K * MAX_RESP_LEN + (K-1) * MAX_PROMPT_LEN
# 默认 K=3 → 3*8192 + 2*20480 = 65536（足够用；可用 CHAIN_RESP_LEN= 覆盖）。
if [[ -z "${CHAIN_RESP_LEN:-}" ]]; then
  if (( K > 1 )); then
    CHAIN_RESP_LEN=$((K * MAX_RESP_LEN + (K - 1) * MAX_PROMPT_LEN))
  else
    CHAIN_RESP_LEN="${MAX_RESP_LEN}"
  fi
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stharc_grpo_k${K}}"

# dynamic bsz 的每 GPU token budget：必须 >= 训练侧最长序列（prompt + chain response）
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LEN + CHAIN_RESP_LEN))}"

# STHARC_GAMMA / LAM / ATTRIBUTION / RUN_FROZEN / BASE_USER_ID
# 已由 _research_workspace_paths.sh 设定（可 env 覆盖）

# ── 目录准备 ──────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}" "${SESSION_DIR}" "${CKPT_DIR}" \
         "${SWANLAB_LOG_DIR}" "${ROLLOUT_DATA_DIR}" \
         "${GSB_JUDGE_IO_LOG_DIR}" "${ACCURACY_JUDGE_IO_LOG_DIR}" \
         "${LOG_DIR}"

# PYTHONPATH 已由 _research_workspace_paths.sh 设置（当前副本根 + 副本内 verl + ai2c + WS_ROOT）
# 注意：启动 python 时勿覆盖掉 WS_ROOT（agentic_train_offline 依赖它）
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR}"
export GSB_JUDGE_IO_LOG_DIR="${GSB_JUDGE_IO_LOG_DIR}"
export ACCURACY_JUDGE_IO_LOG_DIR="${ACCURACY_JUDGE_IO_LOG_DIR}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-Claude-Sonnet-4.6}"
export MAIN_BRAIN_PROMPT_STYLE="${MAIN_BRAIN_PROMPT_STYLE:-sft}"

# reward：对齐生产 pairwise（AiToCCombinedReward）
#   ① 格式门 + 完成门；② GSB 正反双跑 4 维；③ use_accuracy=false（暂关）
#   ④ waste_penalty_lambda=0（efficiency 暂关）；无 baseline → NaN → GRPO mask
# 数据要求：task_chain 每个 task 需含 baseline_answer（造数时从 query/extra_info 透传）
REWARD_COMPONENT_CFG="{_target_: recipe.agentic.reward.components.AiToCCombinedReward, weight: 1.0, use_completion_gate: true, use_step_score: false, target_steps: 3, max_steps: 7, reverse: true, no_baseline_fallback: nan, judge_max_tokens: 512, use_accuracy: false, acc_max_tokens: 8192, acc_max_traj_chars: 10000, acc_default_on_error: 0.5, judge_timeout: 240, judge_retry_delay: 3.0, waste_penalty_lambda: 0, waste_penalty_cap: 3, judge_io_log_dir: '${GSB_JUDGE_IO_LOG_DIR}'}"
LOG_FILE="${LOG_DIR}/stharc_${RUN_ID}.log"

# ── 业务侧 harness 开关 ──────────────────────────────────────────────
# skill_manage：开启后模型可调用 skill_manage 工具，HarnessTracker 追踪写事件
export AI_TO_C_SKILL_MANAGE_ENABLED="${AI_TO_C_SKILL_MANAGE_ENABLED:-1}"
# memory：开启 memory backend 和工具
export AI_TO_C_MEMORY_ENABLED="${AI_TO_C_MEMORY_ENABLED:-1}"
export AI_TO_C_MEMORY_TOOL_ENABLED="${AI_TO_C_MEMORY_TOOL_ENABLED:-1}"
# 沙箱 log：默认只记 skill 清单 + 本 chain 变更 skill 全文 + 最终 memory；
# 设 STHARC_LOG_FULL_SKILLS=1 可把沙箱内全部 SKILL.md 写入 traces JSONL（体积大）
# export STHARC_LOG_FULL_SKILLS=1
# ReAct 每步最大 token 数：RL 训练时压缩为 1024，防止多轮上下文超出 max_model_len
# （生产默认 4096，训练时小值可大幅减少 vLLM context length 超限问题）
export AI_TO_C_REACT_PLANNER_MAX_TOKENS="${AI_TO_C_REACT_PLANNER_MAX_TOKENS:-1024}"

# ── 清理上次遗留的 harness 沙箱（避免 FileExistsError）──────────────────
echo "[STHARC] 清理 /tmp/rl_stharc/ 旧沙箱..."
rm -rf /tmp/rl_stharc/

# ── 数据检查（train 缺失则报错；val 缺失时回退到 train）─────────────────
if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "[STHARC] 错误：train.parquet 不存在，请先离线生成并上传到 ${DATA_DIR}/"
    echo "[STHARC] 默认 DATA_DIR=research/data/stharc_2w（--G 1；GRPO 的 G 由 rollout.n 控制）"
    echo "[STHARC] 离线生成命令示例（在本地 training_code_public 目录执行）："
    echo "  python3 -m recipe.agentic.data.prepare_dataset stharc \\"
    echo "      --train-input /path/to/train.jsonl \\"
    echo "      --train-output ../data/stharc_2w/train.parquet \\"
    echo "      --val-output ../data/stharc_2w/val.parquet \\"
    echo "      --val-ratio 0.05 --K ${K} --G 1 --strategy random --seed 42"
    exit 1
fi

# val 文件若不存在则回退到 train（避免训练启动时 crash）
_VAL_FILE="${DATA_DIR}/val.parquet"
if [[ ! -f "${_VAL_FILE}" ]]; then
    echo "[STHARC] 警告：val.parquet 不存在，val 回退使用 train.parquet"
    _VAL_FILE="${DATA_DIR}/train.parquet"
fi

# ── 预检：脚本 K 与 parquet 全表 task_chain 长度必须一致 ───────────────
echo "[STHARC] 校验数据 K 与脚本 K=${K} 一致（全表）..."
python3 - "${DATA_DIR}/train.parquet" "${_VAL_FILE}" "${K}" <<'PY'
import json
import sys

import pandas as pd

train_path, val_path, k_str = sys.argv[1], sys.argv[2], sys.argv[3]
k = int(k_str)


def chain_lengths(path: str) -> tuple[set[int], int]:
    df = pd.read_parquet(path, columns=["task_chain"])
    lengths: set[int] = set()
    n_rows = 0
    for idx, raw in enumerate(df["task_chain"]):
        n_rows += 1
        if isinstance(raw, str):
            try:
                tc = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    f"[STHARC] 错误：无法解析 task_chain（{path} row={idx}）: {exc}"
                ) from exc
        else:
            tc = list(raw) if raw is not None else []
        lengths.add(len(tc))
    if not lengths:
        raise SystemExit(f"[STHARC] 错误：{path} 中没有可读的 task_chain")
    return lengths, n_rows


for label, path in (("train", train_path), ("val", val_path)):
    lengths, n_rows = chain_lengths(path)
    if lengths != {k}:
        raise SystemExit(
            f"[STHARC] 错误：{label} parquet 的 task_chain 长度={sorted(lengths)}，"
            f"与脚本 K={k} 不一致（已扫全表 {n_rows} 行）。\n"
            f"  文件: {path}\n"
            f"  请用同一 K 重新造数后再开训，例如：\n"
            f"    python3 -m recipe.agentic.data.prepare_dataset stharc \\\n"
            f"        --train-input /path/to/train.jsonl \\\n"
            f"        --train-output {path} \\\n"
            f"        --K {k} --G 1 --strategy random --seed 42\n"
            f"  或改环境变量 K 对齐现有数据（当前数据 K 看起来是 {sorted(lengths)}）。"
        )
    print(f"[STHARC] {label} K check ok: data K == script K == {k} (all {n_rows} rows)")
PY

echo "============================================================"
echo " STHARC-GRPO 启动 (Qwen3.5-0.8B / ${NUM_GPUS}-GPU / main_ppo)"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "  MODEL_PATH           : ${MODEL_PATH}"
echo "  DATA_DIR             : ${DATA_DIR}"
echo "  CKPT_DIR             : ${CKPT_DIR}"
echo "  NUM_GPUS / TP_SIZE   : ${NUM_GPUS} / ${TP_SIZE}"
echo "  BATCH / ROLLOUT_N    : ${BATCH_SIZE} / ${ROLLOUT_N}"
echo "  NUM_AGENT_WORKERS    : ${NUM_AGENT_WORKERS}"
echo "  VAL_BATCH_SIZE       : ${VAL_BATCH_SIZE}  (aligned to num_workers)"
echo "  MAX_PROMPT/RESP      : ${MAX_PROMPT_LEN} / ${MAX_RESP_LEN}"
echo "  MAX_MODEL_LEN        : ${MAX_MODEL_LEN}"
echo "  MAX_NUM_BATCHED_TOK  : ${MAX_NUM_BATCHED_TOKENS}"
echo "  CHAIN_RESP_LEN       : ${CHAIN_RESP_LEN}  (= K*RESP + (K-1)*PROMPT)"
echo "  PPO_MAX_TOKEN_LEN    : ${PPO_MAX_TOKEN_LEN}  (dynamic_bsz budget)"
echo "  K / G                : ${K} / ${G}"
echo "  EXPERIMENT_NAME      : ${EXPERIMENT_NAME}"
echo "  GAMMA / LAM          : ${STHARC_GAMMA} / ${STHARC_LAM}"
echo "  RUN_FROZEN           : ${STHARC_RUN_FROZEN}"
echo "  TOTAL_STEPS          : ${TOTAL_STEPS}"
echo "  SwanLab              : ${SWANLAB_LOG_DIR}"
echo "  GSB IO logs          : ${GSB_JUDGE_IO_LOG_DIR}"
echo "  Reward               : AiToCCombinedReward (pairwise, accuracy off)"
echo "  Judge                : ${ANTHROPIC_BASE_URL:-http://ai-api.jdcloud.com/anthropic/v1}  (${ANTHROPIC_MODEL})"
echo "  PYTHONPATH           : ${PYTHONPATH}"
echo "============================================================"

# ── 预检：确保 Hydra 能 import 到 research 侧 STHARCAgentLoop + CombinedReward ─
python3 - <<'PY'
import os
import recipe.agentic.agent_loop as m
from recipe.agentic.stharc.agent_loop import STHARCAgentLoop
from recipe.agentic.stharc.harness_state import HarnessState
import recipe.agentic.reward.components as comp

print(f"[STHARC] agent_loop file : {m.__file__}")
print(f"[STHARC] components file : {comp.__file__}")
print(f"[STHARC] hydra target    : recipe.agentic.stharc.agent_loop.STHARCAgentLoop")
print(f"[STHARC] STHARCAgentLoop : {STHARCAgentLoop}")
print(f"[STHARC] HarnessState    : {HarnessState}")
from pathlib import Path
import inspect
repo_root = Path(os.environ["VERL_RESEARCH_DIR"]).resolve()
for target, relative_path in (
    (m, "recipe/agentic/agent_loop.py"),
    (comp, "recipe/agentic/reward/components.py"),
    (STHARCAgentLoop, "recipe/agentic/stharc/agent_loop.py"),
    (HarnessState, "recipe/agentic/stharc/harness_state.py"),
):
    actual = Path(inspect.getfile(target)).resolve()
    expected = (repo_root / relative_path).resolve()
    if actual != expected:
        raise SystemExit(
            f"[STHARC] 错误：模块来源不属于当前副本：{actual}\n"
            f"期望：{expected}；请检查 PYTHONPATH，确保当前副本优先。"
        )
if not hasattr(m, "STHARCAgentLoop"):
    raise SystemExit(f"[STHARC] 错误：{m.__file__} 中没有 STHARCAgentLoop")
if not hasattr(comp, "AiToCCombinedReward"):
    reward_dir = os.path.dirname(comp.__file__ or "")
    raise SystemExit(
        "[STHARC] 错误：recipe.agentic.reward.components 缺少 AiToCCombinedReward。\n"
        f"  当前 components.py: {comp.__file__}\n"
        "  训练机代码未同步完整 pairwise reward 包。请从本地 workspace 同步以下文件到本机同路径：\n"
        "    recipe/agentic/reward/components.py          (~2367 行，含 AiToCCombinedReward)\n"
        "    recipe/agentic/reward/gsb_pairwise.py\n"
        "    recipe/agentic/reward/gsb_absolute.py\n"
        "    recipe/agentic/reward/completion.py\n"
        "    recipe/agentic/reward/accuracy_trajectory.py\n"
        "    recipe/agentic/reward/weighted.py\n"
        "    recipe/agentic/reward/__init__.py\n"
        f"  快速自检: wc -l {reward_dir}/components.py  # 期望 >= 2000\n"
        f"  以及: ls {reward_dir}/gsb_pairwise.py {reward_dir}/completion.py"
    )
print(f"[STHARC] AiToCCombinedReward : {comp.AiToCCombinedReward}")
print("[STHARC] preflight import ok")
PY

# ── 启动训练（与生产脚本相同模式：main_ppo + 全 CLI 覆盖）────────────
# PYTHONPATH 沿用 _research_workspace_paths.sh（当前副本优先，含 agentic_train_offline）
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
    hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    "trainer.logger=[console,swanlab]" \
    trainer.project_name=stharc_train \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    "+trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}" \
    "+trainer.swanlab.save_dir=${SWANLAB_LOG_DIR}" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${_VAL_FILE}" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${CHAIN_RESP_LEN}" \
    data.trust_remote_code=True \
    data.return_raw_chat=true \
    data.shuffle=true \
    data.filter_overlong_prompts=false \
    +data.data_source=ai_to_c \
    +data.return_raw_prompt=true \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2" \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_LIST_PATH}" \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    reward.reward_model.enable=false \
    \
    +stharc.env._target_=recipe.agentic.envs.AiToCReactEnv \
    "+stharc.env.ai_to_c_repo_path=${AI2C_REPO_PATH}" \
    "+stharc.env.session_dir_override=${SESSION_DIR}" \
    +stharc.env.record_subagent_traces=false \
    "+stharc.vllm_base_url=${VLLM_BASE_URL}" \
    "+stharc.vllm_model_name=${VLLM_MODEL_NAME}" \
    "+stharc.per_rollout_timeout_sec=${PER_ROLLOUT_TIMEOUT}" \
    "+stharc.gamma=${STHARC_GAMMA}" \
    "+stharc.lam=${STHARC_LAM}" \
    "+stharc.attribution=${STHARC_ATTRIBUTION}" \
    "+stharc.run_frozen_chain=${STHARC_RUN_FROZEN}" \
    "+stharc.base_user_id=${STHARC_BASE_USER_ID}" \
    "+stharc.max_task_response_length=${MAX_RESP_LEN}" \
    "+stharc.max_task_context_length=${MAX_PROMPT_LEN}" \
    "+stharc.reward.normalize=false" \
    "+stharc.reward.clip_min=0.0" \
    "+stharc.reward.clip_max=1.0" \
    "+stharc.reward.components=[${REWARD_COMPONENT_CFG}]" \
    "+stharc.apply_chat_template_kwargs.enable_thinking=false" \
    \
    ray_kwargs.ray_init.num_cpus=32 \
    +ray_kwargs.ray_init.runtime_env.env_vars.GSB_JUDGE_IO_LOG_DIR="${GSB_JUDGE_IO_LOG_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ACCURACY_JUDGE_IO_LOG_DIR="${ACCURACY_JUDGE_IO_LOG_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo " Training finished."
echo " Log:         ${LOG_FILE}"
echo " Checkpoints: ${CKPT_DIR}"
echo " SwanLab:     swanlab upload ${SWANLAB_LOG_DIR}"
echo "============================================================"
