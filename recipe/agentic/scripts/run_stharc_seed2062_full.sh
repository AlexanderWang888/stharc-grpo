#!/usr/bin/env bash
# ======================================================================
# 论文训练 seed 2062：STHARC / SAGE / SkillRise / λ=0 / Random-PU
# ----------------------------------------------------------------------
# 同一份 parquet / val / 7 卡 / 满预算，只换训练 SEED=2062。
#   musique   : stharc / sage / skillrise / lambda0 / random_pu
#   textcraft : stharc / sage(b05) / skillrise / lambda0 / random_pu
# TextCraft SAGE 用正文 b05（bonus_coef=0.5）；MuSiQue SAGE 仍是 +1。
#
# 在 training_code_public 目录：
#   bash recipe/agentic/scripts/run_stharc_seed2062_full.sh
# ======================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_RESEARCH_DIR}"

SEED="${SEED:-2062}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_LOG="${SWEEP_LOG:-/mnt/workspace/stharc/runs/seed${SEED}_full_${STAMP}.log}"
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

echo "[seed-full] SEED=${SEED} STAMP=${STAMP}  10 jobs"
echo "[seed-full] log=${SWEEP_LOG}"
{
  echo "SEED=${SEED} STAMP=${STAMP} MODEL_PATH=${MODEL_PATH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
} | tee -a "${SWEEP_LOG}"

JOBS=(
  "musique stharc"
  "musique sage"
  "musique skillrise"
  "musique lambda0"
  "musique random_pu"
  "textcraft stharc"
  "textcraft sage"
  "textcraft skillrise"
  "textcraft lambda0"
  "textcraft random_pu"
)

failed=0
idx=0
n=${#JOBS[@]}
for spec in "${JOBS[@]}"; do
  idx=$((idx + 1))
  ds="${spec%% *}"
  method="${spec##* }"
  unset METHOD RUN_ID STHARC_CREDIT STHARC_LAM STHARC_RUN_FROZEN
  unset STHARC_ATTRIBUTION STHARC_SAGE_BONUS_COEF STHARC_SKILLRISE_GAMMA
  unset FORCE_STHARC_LAM STHARC_SHUFFLE_SEED

  extra=()
  if [[ "${ds}" == "musique" ]]; then
    tri="${SCRIPT_DIR}/run_stharc_triangle_musique.sh"
    case "${method}" in
      stharc) run_id="musique_verbatim_mechlike_stharc_s${SEED}_${STAMP}" ;;
      sage) run_id="musique_verbatim_mechlike_sage_s${SEED}_${STAMP}" ;;
      skillrise) run_id="musique_verbatim_mechlike_skillrise_s${SEED}_${STAMP}" ;;
      lambda0) run_id="musique_verbatim_mechlike_lambda0_s${SEED}_${STAMP}" ;;
      random_pu) run_id="musique_verbatim_mechlike_random_pu_s${SEED}_${STAMP}" ;;
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
      lambda0) run_id="textcraft_mechlike_lambda0_s${SEED}_${STAMP}" ;;
      random_pu) run_id="textcraft_mechlike_random_pu_s${SEED}_${STAMP}" ;;
    esac
  fi

  echo ""
  echo "[seed-full] ===== ${idx}/${n}  ${ds} METHOD=${method} RUN_ID=${run_id} ====="
  echo "[seed-full] ${idx}/${n} ${ds} ${method} ${run_id} START $(date -Is)" | tee -a "${SWEEP_LOG}"

  if env "${extra[@]}" \
      METHOD="${method}" \
      RUN_ID="${run_id}" \
      bash "${tri}"
  then
    echo "[seed-full] ${idx}/${n} ${ds} ${method} OK $(date -Is)" | tee -a "${SWEEP_LOG}"
  else
    rc=$?
    failed=$((failed + 1))
    echo "[seed-full] ${idx}/${n} ${ds} ${method} FAIL rc=${rc} $(date -Is)" | tee -a "${SWEEP_LOG}"
  fi
  sleep 15
done

echo "[seed-full] done failed=${failed}/${n}  log=${SWEEP_LOG}" | tee -a "${SWEEP_LOG}"
exit "${failed}"
