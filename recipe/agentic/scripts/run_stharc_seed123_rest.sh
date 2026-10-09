#!/usr/bin/env bash
# ======================================================================
# seed 123 补跑：TextCraft SkillRise（被停的那条）+ 两边 λ=0 / Random-PU
# 已完成不用重跑：MuSiQue STHARC/SAGE/SkillRise、TextCraft STHARC/SAGE(b05)
# ======================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_RESEARCH_DIR}"

SEED="${SEED:-123}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_LOG="${SWEEP_LOG:-/mnt/workspace/stharc/runs/seed${SEED}_rest_${STAMP}.log}"
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

echo "[seed-rest] SEED=${SEED} STAMP=${STAMP}  5 jobs"
echo "[seed-rest] log=${SWEEP_LOG}"
{
  echo "SEED=${SEED} STAMP=${STAMP}"
  echo "jobs: textcraft/skillrise musique/lambda0 musique/random_pu textcraft/lambda0 textcraft/random_pu"
} | tee -a "${SWEEP_LOG}"

JOBS=(
  "textcraft skillrise"
  "musique lambda0"
  "musique random_pu"
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

  if [[ "${ds}" == "musique" ]]; then
    tri="${SCRIPT_DIR}/run_stharc_triangle_musique.sh"
    case "${method}" in
      skillrise) run_id="musique_verbatim_mechlike_skillrise_s${SEED}_${STAMP}" ;;
      lambda0) run_id="musique_verbatim_mechlike_lambda0_s${SEED}_${STAMP}" ;;
      random_pu) run_id="musique_verbatim_mechlike_random_pu_s${SEED}_${STAMP}" ;;
    esac
  else
    tri="${SCRIPT_DIR}/run_stharc_triangle_textcraft.sh"
    case "${method}" in
      skillrise) run_id="textcraft_mechlike_skillrise_s${SEED}_${STAMP}" ;;
      lambda0) run_id="textcraft_mechlike_lambda0_s${SEED}_${STAMP}" ;;
      random_pu) run_id="textcraft_mechlike_random_pu_s${SEED}_${STAMP}" ;;
    esac
  fi

  echo ""
  echo "[seed-rest] ===== ${idx}/${n}  ${ds} METHOD=${method} RUN_ID=${run_id} ====="
  echo "[seed-rest] ${idx}/${n} ${ds} ${method} ${run_id} START $(date -Is)" | tee -a "${SWEEP_LOG}"

  if METHOD="${method}" RUN_ID="${run_id}" bash "${tri}"
  then
    echo "[seed-rest] ${idx}/${n} ${ds} ${method} OK $(date -Is)" | tee -a "${SWEEP_LOG}"
  else
    rc=$?
    failed=$((failed + 1))
    echo "[seed-rest] ${idx}/${n} ${ds} ${method} FAIL rc=${rc} $(date -Is)" | tee -a "${SWEEP_LOG}"
  fi
  sleep 15
done

echo "[seed-rest] done failed=${failed}/${n}  log=${SWEEP_LOG}" | tee -a "${SWEEP_LOG}"
exit "${failed}"
