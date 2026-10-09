#!/usr/bin/env bash
# ======================================================================
# AppWorld oracle SFT — 已切到 LLaMA-Factory（与 diagnostic SFT 同管线）
# ======================================================================
# 数据生成（在 research/ 下）:
#   python3 dataset_explore/idea2_datasets/gen_sft_oracle_appworld.py \
#     --appworld-root appworld_explore \
#     --out-dir training_code_public/data/stharc_appworld_sft \
#     --difficulty-max 2 --mode stepwise --with-format-few-shot
#
# 训练（训练机，LlamaFactory-new）:
#   rsync data/stharc_appworld_sft + dataset_info + yaml/sh 后:
#   cd /mnt/workspace/agent_train/ai2c_workspace/LlamaFactory-new
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-2B \
#     bash examples/train_full/run_stharc_appworld_sft.sh
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-4B \
#     bash examples/train_full/run_stharc_appworld_sft.sh
#   MODEL_PATH=/mnt/workspace/ckpt/Qwen3.5-9B \
#     bash examples/train_full/run_stharc_appworld_sft.sh
#
# 本脚本只做转发，避免误跑旧 verl sft_trainer。
# ======================================================================
set -euo pipefail

LF_ROOT="${LF_ROOT:-}"
if [[ -z "${LF_ROOT}" ]]; then
  # 本机 workspace / 训练机常见布局
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CANDIDATES=(
    "${HERE}/../../../../../LlamaFactory-new"
    "/mnt/workspace/agent_train/ai2c_workspace/LlamaFactory-new"
  )
  for c in "${CANDIDATES[@]}"; do
    if [[ -f "${c}/examples/train_full/run_stharc_appworld_sft.sh" ]]; then
      LF_ROOT="${c}"
      break
    fi
  done
fi

if [[ -z "${LF_ROOT}" || ! -f "${LF_ROOT}/examples/train_full/run_stharc_appworld_sft.sh" ]]; then
  echo "[sft-appworld] LlamaFactory launcher not found."
  echo "  cd LlamaFactory-new && bash examples/train_full/run_stharc_appworld_sft.sh"
  exit 2
fi

echo "[sft-appworld] forwarding to LLaMA-Factory: ${LF_ROOT}"
cd "${LF_ROOT}"
exec bash examples/train_full/run_stharc_appworld_sft.sh "$@"
