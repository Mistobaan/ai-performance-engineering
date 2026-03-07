#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_gpu_stream_bench.sh [options]

Runs a GPU STREAM-style memory benchmark and writes:
  results/structured/<run_id>_<label>_gpu_stream.json
  results/structured/<run_id>_<label>_gpu_stream.csv
  results/structured/<run_id>_<label>_gpu_stream_clock_lock.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>        Label for output paths (default: hostname)
  --device <n>           CUDA device index (default: 0)
  --size-mb <n>          Vector size in MB (default: 1024)
  --iters <n>            Measured iterations per op (default: 40)
  --warmup <n>           Warmup iterations per op (default: 10)
  --dtype <fp32|fp16|bf16>  Data type (default: fp32)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
LABEL="${LABEL:-$(hostname)}"
DEVICE="${DEVICE:-0}"
SIZE_MB="${SIZE_MB:-1024}"
ITERS="${ITERS:-40}"
WARMUP="${WARMUP:-10}"
DTYPE="${DTYPE:-fp32}"
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --size-mb) SIZE_MB="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_STRUCT_DIR"
OUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_gpu_stream.json"
OUT_CSV="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_gpu_stream.csv"
LOCK_META_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_gpu_stream_clock_lock.json"

# Enforce strict GPU clock locking.
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  export RUN_ID LABEL
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --devices "$DEVICE" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "${ORIG_ARGS[@]}"
fi

echo "== GPU STREAM benchmark =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "DEVICE=${DEVICE}"
echo "SIZE_MB=${SIZE_MB}"
echo "ITERS=${ITERS}"
echo "WARMUP=${WARMUP}"
echo "DTYPE=${DTYPE}"
echo "OUT_JSON=${OUT_JSON}"
echo "OUT_CSV=${OUT_CSV}"

"${ROOT_DIR}/env/venv/bin/python" "${ROOT_DIR}/scripts/torch_gpu_stream_bench.py" \
  --run-id "$RUN_ID" \
  --label "$LABEL" \
  --device "$DEVICE" \
  --size-mb "$SIZE_MB" \
  --iters "$ITERS" \
  --warmup "$WARMUP" \
  --dtype "$DTYPE" \
  --output-json "$OUT_JSON" \
  --output-csv "$OUT_CSV"
