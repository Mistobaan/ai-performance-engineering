#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_gemm_with_telemetry.sh --gpu <idx> [options]

Runs a single-GPU Torch GEMM microbench under strict GPU clock lock and records
1 Hz telemetry via nvidia-smi while the benchmark is running.

Outputs:
  runs/<run_id>/structured/<run_id>_<label>_gpu<idx>_gemm.csv
  runs/<run_id>/structured/<run_id>_<label>_gpu<idx>_gemm_clock_lock.json
  runs/<run_id>/raw/<run_id>_<label>_gpu<idx>_gemm.log
  runs/<run_id>/raw/<run_id>_<label>_gpu<idx>_gemm_telemetry_query.csv
  runs/<run_id>/raw/<run_id>_<label>_gpu<idx>_gemm_telemetry_pmon.log

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD_HHMMSS_gemm_telemetry)
  --label <label>        Label prefix (default: hostname)
  --gpu <idx>            Physical GPU index to run (required)
  --m <int>              GEMM M (default: 16384)
  --n <int>              GEMM N (default: 16384)
  --k <int>              GEMM K (default: 16384)
  --dtype <bf16|fp16>    Data type (default: bf16)
  --iters <n>            Iterations (default: 10000)
  --interval <sec>       Telemetry sampling interval seconds (default: 1)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H%M%S)_gemm_telemetry}"
LABEL="${LABEL:-$(hostname)}"
GPU=""
M="${M:-16384}"
N="${N:-16384}"
K="${K:-16384}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-10000}"
INTERVAL="${INTERVAL:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$GPU" ]]; then
  echo "ERROR: --gpu is required" >&2
  usage
  exit 2
fi

case "$DTYPE" in
  bf16|fp16) ;;
  *) echo "ERROR: --dtype must be bf16 or fp16" >&2; exit 2 ;;
esac

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCT="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW" "$OUT_STRUCT"

# Deterministic output names (fail fast on collision to avoid mixing schemas/runs).
GEMM_CSV="${OUT_STRUCT}/${RUN_ID}_${LABEL}_gpu${GPU}_gemm.csv"
LOCK_META="${OUT_STRUCT}/${RUN_ID}_${LABEL}_gpu${GPU}_gemm_clock_lock.json"
BENCH_LOG="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu${GPU}_gemm.log"
TELE_CSV="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu${GPU}_gemm_telemetry_query.csv"
PMON_LOG="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu${GPU}_gemm_telemetry_pmon.log"

for p in "$GEMM_CSV" "$LOCK_META" "$BENCH_LOG" "$TELE_CSV" "$PMON_LOG"; do
  if [[ -e "$p" ]]; then
    echo "ERROR: output already exists: $p" >&2
    echo "Fix: choose a new --run-id (recommended) or --label." >&2
    exit 2
  fi
done

QUERY_FIELDS=(
  timestamp
  index
  utilization.gpu
  utilization.memory
  clocks.current.sm
  clocks.current.memory
  power.draw
  temperature.gpu
  clocks_event_reasons.sw_power_cap
  clocks_event_reasons.hw_thermal_slowdown
  clocks_event_reasons.hw_power_brake_slowdown
  clocks_event_reasons.sw_thermal_slowdown
)
QUERY_CSV="--query-gpu=$(IFS=','; echo "${QUERY_FIELDS[*]}")"

echo "== GEMM + Telemetry =="
echo "run_id=${RUN_ID}"
echo "label=${LABEL}"
echo "gpu=${GPU}"
echo "m,n,k=${M},${N},${K}"
echo "dtype=${DTYPE}"
echo "iters=${ITERS}"
echo "interval_s=${INTERVAL}"
echo
echo "Outputs:"
echo "  ${GEMM_CSV}"
echo "  ${LOCK_META}"
echo "  ${BENCH_LOG}"
echo "  ${TELE_CSV}"
echo "  ${PMON_LOG}"
echo

# Telemetry header row (keep units; plotter expects nvidia-smi's default header keys).
nvidia-smi $QUERY_CSV --format=csv >"$TELE_CSV" || true

set +e
CUDA_VISIBLE_DEVICES="${GPU}" RUN_ID="${RUN_ID}" LABEL="${LABEL}_gpu${GPU}" \
  "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" --lock-meta-out "$LOCK_META" -- \
  "${ROOT_DIR}/env/venv/bin/python" "${ROOT_DIR}/scripts/torch_gemm_bench.py" \
  --m "${M}" --n "${N}" --k "${K}" \
  --dtype "${DTYPE}" --iters "${ITERS}" \
  --label "${LABEL}_gpu${GPU}" \
  --output-csv "$GEMM_CSV" >"$BENCH_LOG" 2>&1 &
BENCH_PID=$!

while kill -0 "$BENCH_PID" >/dev/null 2>&1; do
  nvidia-smi $QUERY_CSV --format=csv,noheader >>"$TELE_CSV" || true
  nvidia-smi pmon -c 1 >>"$PMON_LOG" || true
  sleep "$INTERVAL"
done

wait "$BENCH_PID"
BENCH_RC=$?
set -e

echo "bench_rc=${BENCH_RC}"
exit "$BENCH_RC"
