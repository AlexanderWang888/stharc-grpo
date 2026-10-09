#!/usr/bin/env bash
# ======================================================================
# CreditAlign 测评（M1 / E1）：消费 STHARC chain traces，输出一致率报告
# ======================================================================
# 用法：
#   # 1) 本地无 GPU：用 mech parquet 合成理想 agent traces（task 图匹配）
#   bash recipe/agentic/scripts/run_credit_align_eval.sh --demo
#
#   # 2) 对真实训练 traces 做 CreditAlign（自由命名 OK）
#   TRACE_DIR=/mnt/workspace/stharc/runs/public/rollout_logs/<run>/traces \
#   PARQUET=data/stharc_diagnostic_mech/train.parquet \
#     bash recipe/agentic/scripts/run_credit_align_eval.sh
#
#   # 3) 指定 parquet 做 demo
#   DEMO_PARQUET=data/stharc_diagnostic_mech/val.parquet \
#     bash recipe/agentic/scripts/run_credit_align_eval.sh --demo
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_RESEARCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESEARCH_DIR="$(cd "${VERL_RESEARCH_DIR}/.." && pwd)"
DATASET_EXPLORE_DIR="$(cd "${VERL_RESEARCH_DIR}/../dataset_explore" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

STHARC_HOME="${STHARC_HOME:-/mnt/workspace/stharc}"
OUT_DIR="${OUT_DIR:-${STHARC_HOME}/runs/credit_align}"
mkdir -p "${OUT_DIR}"

GAMMA="${STHARC_GAMMA:-0.9}"
ATTRIBUTION="${STHARC_ATTRIBUTION:-gamma_decay}"

if [[ "${1:-}" == "--demo" ]]; then
    DEMO_PARQUET="${DEMO_PARQUET:-${VERL_RESEARCH_DIR}/data/stharc_diagnostic_mech/val.parquet}"
    echo "[CreditAlign] demo mode from ${DEMO_PARQUET}"
    "${PYTHON_BIN}" "${DATASET_EXPLORE_DIR}/verify_credit_from_rollouts.py" \
        --demo-from-parquet "${DEMO_PARQUET}" \
        --demo-out-dir "${OUT_DIR}/demo_traces" \
        --gamma "${GAMMA}" \
        --attribution "${ATTRIBUTION}" \
        --out-md "${OUT_DIR}/credit_align_demo.md" \
        --out-json "${OUT_DIR}/credit_align_demo.json"
    echo "[CreditAlign] report: ${OUT_DIR}/credit_align_demo.md"
    exit 0
fi

TRACE_DIR="${TRACE_DIR:-}"
if [[ -z "${TRACE_DIR}" ]]; then
    echo "ERROR: set TRACE_DIR to rollout traces/ directory, or pass --demo"
    echo "  TRACE_DIR=/path/to/traces bash $0"
    echo "  bash $0 --demo"
    exit 2
fi

PARQUET_ARGS=()
if [[ -n "${PARQUET:-}" ]]; then
    PARQUET_ARGS+=(--parquet "${PARQUET}")
    echo "[CreditAlign] will backfill provenance from ${PARQUET}"
fi

echo "[CreditAlign] evaluating traces under ${TRACE_DIR}"
"${PYTHON_BIN}" "${DATASET_EXPLORE_DIR}/verify_credit_from_rollouts.py" \
    --trace-dir "${TRACE_DIR}" \
    --gamma "${GAMMA}" \
    --attribution "${ATTRIBUTION}" \
    "${PARQUET_ARGS[@]}" \
    --out-md "${OUT_DIR}/credit_align_report.md" \
    --out-json "${OUT_DIR}/credit_align_report.json"
echo "[CreditAlign] report: ${OUT_DIR}/credit_align_report.md"
