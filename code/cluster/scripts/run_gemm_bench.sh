#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_gemm_bench.sh [options]

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --sizes <m,...>        Comma-separated square GEMM sizes (default: 8192,16384)
  --dtype <bf16|fp16>    Data type (default: bf16)
  --iters <n>            Iterations per size (default: 10)
  --label <label>        Label for CSV (default: hostname)
  --output <path>        Output CSV (default: results/structured/<run_id>_gemm.csv)
  -h, --help             Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"
RUN_ID="$(date +%Y-%m-%d)"
SIZES="8192,16384"
DTYPE="bf16"
ITERS=10
LABEL="$(hostname)"
OUTPUT=""
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --sizes)
      SIZES="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
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

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  export RUN_ID LABEL
  LOCK_META_OUT="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}/${RUN_ID}_${LABEL}_gemm_clock_lock.json"
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "${ORIG_ARGS[@]}"
fi

OUT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_DIR"

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${OUT_DIR}/${RUN_ID}_gemm.csv"
fi

PY="${ROOT_DIR}/env/venv/bin/python"
SCRIPT="${ROOT_DIR}/scripts/torch_gemm_bench.py"

IFS=',' read -r -a SIZE_ARR <<<"$SIZES"
for size in "${SIZE_ARR[@]}"; do
  if [[ -z "$size" ]]; then
    continue
  fi
  "$PY" "$SCRIPT" \
    --m "$size" \
    --n "$size" \
    --k "$size" \
    --dtype "$DTYPE" \
    --iters "$ITERS" \
    --label "$LABEL" \
    --output-csv "$OUTPUT"
done

echo "Wrote ${OUTPUT}"
