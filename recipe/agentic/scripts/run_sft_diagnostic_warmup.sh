#!/usr/bin/env bash
# ======================================================================
# Diagnostic harness SFT 冷启动（memory + skill_modify oracle）
# ======================================================================
# 用法（在 training_code_public 目录）：
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#   NPROC=8 TOTAL_EPOCHS=2 \
#   bash recipe/agentic/scripts/run_sft_diagnostic_warmup.sh
#
# 数据默认：data/stharc_diagnostic_sft/{train,val}.parquet
# 产出 ckpt 后，GRPO 用同一 MODEL_PATH 指到该 ckpt 再跑三角。
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_DIR}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-data/stharc_diagnostic_sft}"
SAVE_PATH="${SAVE_PATH:-/mnt/workspace/stharc/runs/public/checkpoints/stharc_sft_diagnostic_warmup_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
LR="${LR:-1e-5}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
PROJECT_NAME="${PROJECT_NAME:-stharc_sft_diagnostic}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-harness_warmup_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
  echo "[sft] missing ${DATA_DIR}/train.parquet — generate first:"
  echo "  cd .. && python3 dataset_explore/idea2_datasets/gen_sft_oracle_diagnostic.py \\"
  echo "    --out-dir training_code_public/data/stharc_diagnostic_sft --n-per-source 24 --seed 4242"
  exit 2
fi

# 防止误用 GRPO parquet（无 messages 列，会 KeyError）
python3 - <<PY
import pandas as pd, sys
p = "${DATA_DIR}/train.parquet"
cols = list(pd.read_parquet(p).columns)
print("[sft] columns:", cols)
if "messages" not in cols:
    print("[sft] ERROR: need messages column. Got GRPO-style data?", p)
    print("[sft] Use DATA_DIR=data/stharc_diagnostic_sft  (not stharc_appworld_smoke / stharc_diagnostic_mech)")
    sys.exit(2)
print("[sft] rows:", len(pd.read_parquet(p)))
PY

mkdir -p "${SAVE_PATH}"
echo "[sft] model=${MODEL_PATH}"
echo "[sft] data=${DATA_DIR}"
echo "[sft] save=${SAVE_PATH}"
echo "[sft] epochs=${TOTAL_EPOCHS} nproc=${NPROC}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.messages_key=messages \
  data.max_length="${MAX_LENGTH}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  optim.lr="${LR}" \
  engine=fsdp \
  model.path="${MODEL_PATH}" \
  model.use_remove_padding=true \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='["console"]' \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"

echo "[sft] done. ckpt under ${SAVE_PATH}"
echo "[sft] next: GRPO with MODEL_PATH=${SAVE_PATH}/..."
