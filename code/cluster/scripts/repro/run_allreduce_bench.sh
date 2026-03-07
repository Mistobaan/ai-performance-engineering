#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the PyTorch all-reduce bandwidth benchmark (table + busbw-None.png + JSON).

This reproduces the "The average bandwidth of all_reduce..." screenshot.

Usage:
  scripts/repro/run_allreduce_bench.sh [--run-id <id>] [--nproc <n>] [--label <label>]
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="$(hostname)"
NPROC=""
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$NPROC" ]]; then
  NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

# Enforce strict GPU clock locking for the whole torchrun job.
export RUN_ID LABEL
resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$STRUCT_DIR"
LOCK_META_OUT="${STRUCT_DIR}/${RUN_ID}_${LABEL}_allreduce_bench_clock_lock.json"
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "${ORIG_ARGS[@]}"
fi

TORCHRUN="${ROOT_DIR}/env/venv/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
  echo "Missing torchrun at ${TORCHRUN}" >&2
  exit 1
fi

BENCH_SCRIPT="${ROOT_DIR}/tools/all_reduce_bench.py"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Missing benchmark script at ${BENCH_SCRIPT}" >&2
  exit 1
fi

OUT_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}/${RUN_ID}_${LABEL}_allreduce_bench"
mkdir -p "$OUT_DIR"

LOG_PATH="${OUT_DIR}/run.log"

echo "== all-reduce bench =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "NPROC=${NPROC}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_PATH=${LOG_PATH}"

(
  set -x
  cd "$OUT_DIR"
  "$TORCHRUN" \
    --nproc_per_node "$NPROC" \
    --rdzv_endpoint localhost:29500 \
    --rdzv_backend c10d \
    "$BENCH_SCRIPT"
) 2>&1 | tee "$LOG_PATH"

# Copy the latest JSON output into the run-local structured directory for easy discovery.
latest_json=""
if compgen -G "${OUT_DIR}/logs/allreduce_benchmark_results_"'*'.json > /dev/null; then
  latest_json="$(ls -t "${OUT_DIR}/logs/allreduce_benchmark_results_"*.json | head -n 1)"
fi

if [[ -n "$latest_json" ]]; then
  struct_out="${STRUCT_DIR}/${RUN_ID}_${LABEL}_allreduce_benchmark_results.json"
  cp -f "$latest_json" "$struct_out"
  echo "Wrote ${struct_out}"
else
  echo "WARNING: no logs/allreduce_benchmark_results_*.json found under ${OUT_DIR}" >&2
fi
