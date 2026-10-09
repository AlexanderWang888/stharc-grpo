#!/usr/bin/env bash
# ======================================================================
# 论文第二 seed：STHARC + SAGE + SkillRise × MuSiQue + TextCraft
# ----------------------------------------------------------------------
# 同一份 parquet / val / 7 卡 / 满预算。只换训练 SEED=123。
# 现有主表当 seed A；本脚本是 seed 123（6 条，顺序跑）。
# TextCraft SAGE 用正文 b05（bonus_coef=0.5）；MuSiQue SAGE 仍是 +1。
#
# 在 training_code_public 目录：
#   bash recipe/agentic/scripts/run_stharc_seed123_sota.sh
# ======================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_RESEARCH_DIR}"

SEED="${SEED:-123}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_LOG="${SWEEP_LOG:-/mnt/workspace/stharc/runs/seed${SEED}_sota_${STAMP}.log}"
mkdir -p "$(dirname "${SWEEP_LOG}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,5,6,7}"
export MODEL_PATH="${MODEL_PATH:-/mnt/workspace/ckpt/Qwen3.5-9B}"
export SEED
export NUM_GPUS=7
export TP_SIZE=1
export BATCH_SIZE=1
export ROLLOUT_N=7
export NUM_AGENT_WORKERS=7
export VAL_BATCH_SIZE=7
export TOTAL_STEPS=200
export SAVE_FREQ=20
export TEST_FREQ=20
export RESUME_MODE=disable
export GPU_MEM_UTIL=0.40
export MAX_NUM_SEQS=16
unset EVAL_ONLY VAL_ONLY ALLOW_ENV_OVERRIDE
unset CHAIN_RESP_LEN PPO_MAX_TOKEN_LEN K_MAX

echo "[seed-sweep] SEED=${SEED} STAMP=${STAMP}"
echo "[seed-sweep] log=${SWEEP_LOG}"
echo "[seed-sweep] 6 jobs: musique/{stharc,sage,skillrise} then textcraft/{stharc,sage,skillrise}"
{
  echo "SEED=${SEED} STAMP=${STAMP} MODEL_PATH=${MODEL_PATH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
} | tee -a "${SWEEP_LOG}"

JOBS=(
  "musique stharc"
  "musique sage"
  "musique skillrise"
  "textcraft stharc"
  "textcraft sage"
  "textcraft skillrise"
)

failed=0
idx=0
for spec in "${JOBS[@]}"; do
  idx=$((idx + 1))
  ds="${spec%% *}"
  method="${spec##* }"
  unset METHOD RUN_ID STHARC_CREDIT STHARC_LAM STHARC_RUN_FROZEN
  unset STHARC_ATTRIBUTION STHARC_SAGE_BONUS_COEF STHARC_SKILLRISE_GAMMA
  unset FORCE_STHARC_LAM

  extra=()
  if [[ "${ds}" == "musique" ]]; then
    tri="${SCRIPT_DIR}/run_stharc_triangle_musique.sh"
    case "${method}" in
      stharc) run_id="musique_verbatim_mechlike_stharc_s${SEED}_${STAMP}" ;;
      sage) run_id="musique_verbatim_mechlike_sage_s${SEED}_${STAMP}" ;;
      skillrise) run_id="musique_verbatim_mechlike_skillrise_s${SEED}_${STAMP}" ;;
    esac
  else
    tri="${SCRIPT_DIR}/run_stharc_triangle_textcraft.sh"
    case "${method}" in
      stharc) run_id="textcraft_mechlike_stharc_s${SEED}_${STAMP}" ;;
      sage)
        run_id="textcraft_mechlike_sage_b05_s${SEED}_${STAMP}"
        extra+=(STHARC_SAGE_BONUS_COEF=0.5)
        ;;
      skillrise) run_id="textcraft_mechlike_skillrise_s${SEED}_${STAMP}" ;;
    esac
  fi

  echo ""
  echo "[seed-sweep] ===== ${idx}/6  ${ds} METHOD=${method} RUN_ID=${run_id} ====="
  echo "[seed-sweep] ${idx}/6 ${ds} ${method} ${run_id} START $(date -Is)" | tee -a "${SWEEP_LOG}"

  if env "${extra[@]}" \
      METHOD="${method}" \
      RUN_ID="${run_id}" \
      bash "${tri}"
  then
    echo "[seed-sweep] ${idx}/6 ${ds} ${method} OK $(date -Is)" | tee -a "${SWEEP_LOG}"
  else
    rc=$?
    failed=$((failed + 1))
    echo "[seed-sweep] ${idx}/6 ${ds} ${method} FAIL rc=${rc} $(date -Is)" | tee -a "${SWEEP_LOG}"
  fi
  sleep 15
done

echo "[seed-sweep] done failed=${failed}/6  log=${SWEEP_LOG}" | tee -a "${SWEEP_LOG}"
exit "${failed}"
