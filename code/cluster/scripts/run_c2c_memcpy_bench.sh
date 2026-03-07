#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_c2c_memcpy_bench.sh [options]

Measures CPU<->GPU memcpy bandwidth + latency for host memory types:
  - pageable
  - pinned
  - managed (cudaMallocManaged)

Outputs:
  results/structured/<run_id>_<label>_c2c_memcpy.json
  results/structured/<run_id>_<label>_c2c_memcpy_clock_lock.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>        Label for output paths (default: hostname)
  --device <n>           CUDA device index (default: 0)
  --bw-sizes <csv>       Bandwidth sizes in bytes (default: 4194304,67108864,1073741824)
  --lat-sizes <csv>      Latency sizes in bytes (default: 4,4096,65536)
  --bw-iters <n>         Bandwidth iterations (default: 20)
  --lat-iters <n>        Latency iterations (default: 20000)
  --warmup <n>           Warmup iterations (default: 5)
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
BW_SIZES="${BW_SIZES:-4194304,67108864,1073741824}"
LAT_SIZES="${LAT_SIZES:-4,4096,65536}"
BW_ITERS="${BW_ITERS:-20}"
LAT_ITERS="${LAT_ITERS:-20000}"
WARMUP="${WARMUP:-5}"

ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --bw-sizes) BW_SIZES="$2"; shift 2 ;;
    --lat-sizes) LAT_SIZES="$2"; shift 2 ;;
    --bw-iters) BW_ITERS="$2"; shift 2 ;;
    --lat-iters) LAT_ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_STRUCT_DIR"
OUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_c2c_memcpy.json"
LOCK_META_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_c2c_memcpy_clock_lock.json"

# Enforce strict GPU clock locking.
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  export RUN_ID LABEL
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --devices "$DEVICE" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "${ORIG_ARGS[@]}"
fi

SRC="${ROOT_DIR}/scripts/benchmarks/c2c_memcpy_bench.cu"
BIN_DIR="/tmp/aisp_cluster_bench"
BIN="${BIN_DIR}/c2c_memcpy_bench"
mkdir -p "$BIN_DIR"

detect_nvcc() {
  # Prefer the toolkit path directly; on some images, /usr/local/bin/nvcc
  # resolves includes relative to /usr/local/bin and fails to find cuda headers.
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    echo "/usr/local/cuda/bin/nvcc"
    return 0
  fi
  command -v nvcc >/dev/null 2>&1 || return 1
  command -v nvcc
}

NVCC_BIN="$(detect_nvcc || true)"
if [[ -z "$NVCC_BIN" ]]; then
  echo "ERROR: nvcc not found; cannot build c2c memcpy bench." >&2
  exit 1
fi

CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC_EXTRA_ARGS=()
for inc_dir in \
  "${CUDA_ROOT}/include" \
  "${CUDA_ROOT}/targets/sbsa-linux/include" \
  "${CUDA_ROOT}/targets/x86_64-linux/include"; do
  if [[ -d "$inc_dir" ]]; then
    NVCC_EXTRA_ARGS+=(-I "$inc_dir")
  fi
done

detect_sm_arch() {
  local dev="$1"
  "${ROOT_DIR}/env/venv/bin/python" - <<'PY' "$dev"
import torch
import sys

dev = int(sys.argv[1])
maj, minor = torch.cuda.get_device_capability(dev)
print(f"sm_{maj}{minor}")
PY
}

if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  arch="$(detect_sm_arch "$DEVICE")"
  echo "Building ${BIN} (arch=${arch}, nvcc=${NVCC_BIN})..."
  "$NVCC_BIN" -O3 -std=c++17 -lineinfo -arch="${arch}" "${NVCC_EXTRA_ARGS[@]}" -o "$BIN" "$SRC"
fi

echo "== CPU<->GPU memcpy (C2C) =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "DEVICE=${DEVICE}"
echo "BW_SIZES=${BW_SIZES}"
echo "LAT_SIZES=${LAT_SIZES}"
echo "BW_ITERS=${BW_ITERS}"
echo "LAT_ITERS=${LAT_ITERS}"
echo "WARMUP=${WARMUP}"
echo "OUT=${OUT_JSON}"
echo

"$BIN" \
  --run-id "$RUN_ID" \
  --label "$LABEL" \
  --device "$DEVICE" \
  --bw-sizes "$BW_SIZES" \
  --lat-sizes "$LAT_SIZES" \
  --bw-iters "$BW_ITERS" \
  --lat-iters "$LAT_ITERS" \
  --warmup "$WARMUP" >"$OUT_JSON"

echo "Wrote ${OUT_JSON}"
