#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_mamf_finder_all_nodes.sh --hosts <h1,h2,...> [options]

Runs the MAMF (Maximum Achievable Matmul FLOPS) finder on each GPU of each
host to find the TRUE achievable TFLOPS ceiling and detect GPU stragglers.

This is the single most important compute diagnostic: it tells you the REAL
performance bar (not theoretical) so you know when to stop optimizing.

Unlike the fixed-shape GEMM sanity check, this scans many matmul shapes to
find the peak achievable TFLOPS, revealing tile/wave quantization effects
unique to each GPU.

Outputs (per host):
  runs/<run_id>/structured/<run_id>_<label>_mamf.csv       (all shapes)
  runs/<run_id>/structured/<run_id>_<label>_mamf_summary.json (best shape + stats)

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs on driver)
  --dtype <bf16|fp16>    Data type (default: bf16)
  --mode <quick|medium|thorough>  Scan thoroughness (default: medium)
  --concurrent           Run all GPUs on a node simultaneously (straggler detection)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
DTYPE="${DTYPE:-bf16}"
MODE="medium"
CONCURRENT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --concurrent) CONCURRENT=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot auto-detect --gpus-per-node" >&2
    exit 2
  fi
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

# Scan ranges by mode
case "$MODE" in
  quick)
    M_RANGE="256 20480 2048"
    N_FIXED="4096"
    K_FIXED="4096"
    ITERS=20
    WARMUP=5
    ;;
  medium)
    M_RANGE="1024 16384 1024"
    N_RANGE="1024 16384 1024"
    K_RANGE="1024 16384 1024"
    ITERS=50
    WARMUP=10
    ;;
  thorough)
    M_RANGE="256 20480 512"
    N_RANGE="256 20480 512"
    K_RANGE="256 20480 512"
    ITERS=100
    WARMUP=20
    ;;
  *)
    echo "ERROR: --mode must be quick, medium, or thorough" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=5
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
LOCAL_STRUCTURED_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
REMOTE_STRUCTURED_DIR="$(cluster_structured_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
mkdir -p "$LOCAL_STRUCTURED_DIR"

run_remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
}

for idx in "${!HOST_ARR[@]}"; do
  host="$(echo "${HOST_ARR[$idx]}" | xargs)"
  [[ -n "$host" ]] || continue
  label=""
  if [[ -n "$LABELS" ]]; then
    label="$(echo "${LABEL_ARR[$idx]}" | xargs)"
  fi
  if [[ -z "$label" ]]; then
    label="$(sanitize_label "$host")"
  fi

  echo "========================================"
  echo "MAMF Finder: host=${host} label=${label} mode=${MODE}"
  echo "========================================"

  if $CONCURRENT; then
    # Run all GPUs simultaneously for straggler detection
    pids=()
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
      out_csv="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_mamf.csv"
      out_json="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_mamf_summary.json"

      # Each worker is constrained to one physical GPU via CUDA_VISIBLE_DEVICES.
      # Inside that namespace the visible device index is always 0.
      mamf_args="scripts/run_with_gpu_clocks.sh --devices 0 -- env/venv/bin/python scripts/mamf_finder.py"

      if [[ "$MODE" == "quick" ]]; then
        mamf_args+=" --m-range '${M_RANGE}' --n ${N_FIXED} --k ${K_FIXED}"
      else
        mamf_args+=" --m-range '${M_RANGE}' --n-range '${N_RANGE}' --k-range '${K_RANGE}'"
      fi
      mamf_args+=" --dtype ${DTYPE} --warmup-iters ${WARMUP} --iters ${ITERS}"
      mamf_args+=" --label ${label}_gpu${gpu} --output-csv ${out_csv} --output-json ${out_json}"

      remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && CUDA_VISIBLE_DEVICES=${gpu} ${mamf_args}"

      echo "  Launching GPU${gpu} (concurrent)..."
      if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
        bash -lc "$remote_cmd" &
        pids+=($!)
      else
        run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")" &
        pids+=($!)
      fi
    done

    # Wait for all GPUs
    echo "  Waiting for ${#pids[@]} concurrent MAMF runs..."
    for pid in "${pids[@]}"; do
      wait "$pid" || echo "WARNING: PID $pid exited with non-zero status" >&2
    done
    echo "  All GPUs complete for ${label}."
  else
    # Sequential per-GPU (safer, less resource contention)
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
      out_csv="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_mamf.csv"
      out_json="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_mamf_summary.json"

      echo "== ${label} GPU${gpu} =="

      mamf_args=(
        scripts/run_with_gpu_clocks.sh
        # CUDA_VISIBLE_DEVICES narrows visibility to one GPU; use local index 0.
        --devices "0"
        --
        env/venv/bin/python
        scripts/mamf_finder.py
      )

      if [[ "$MODE" == "quick" ]]; then
        mamf_args+=(--m-range "${M_RANGE}" --n "${N_FIXED}" --k "${K_FIXED}")
      else
        mamf_args+=(--m-range "${M_RANGE}" --n-range "${N_RANGE}" --k-range "${K_RANGE}")
      fi

      mamf_args+=(
        --dtype "${DTYPE}"
        --warmup-iters "${WARMUP}"
        --iters "${ITERS}"
        --label "${label}_gpu${gpu}"
        --output-csv "${out_csv}"
        --output-json "${out_json}"
      )

      bench_str="$(printf '%q ' "${mamf_args[@]}")"
      remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && CUDA_VISIBLE_DEVICES=${gpu} ${bench_str}"

      if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
        bash -lc "$remote_cmd"
      else
        run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")"
      fi
    done
  fi

  # Fetch results back to the driver for plotting/reporting.
  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
      for suffix in "mamf.csv" "mamf_summary.json"; do
        remote_path="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_${suffix}"
        scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${remote_path}" \
          "${LOCAL_STRUCTURED_DIR}/" 2>/dev/null || true
      done
    done
  fi
done

echo ""
echo "========================================"
echo "MAMF Finder complete."
echo ""
echo "Summary JSONs:"
for idx in "${!HOST_ARR[@]}"; do
  host="$(echo "${HOST_ARR[$idx]}" | xargs)"
  label=""
  if [[ -n "$LABELS" ]]; then
    label="$(echo "${LABEL_ARR[$idx]}" | xargs)"
  fi
  if [[ -z "$label" ]]; then
    label="$(sanitize_label "$host")"
  fi
  for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
    json_path="${LOCAL_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_mamf_summary.json"
    if [[ -f "${json_path}" ]]; then
      mamf=$(python3 -c "import json; d=json.load(open('${json_path}')); print(f\"{d['mamf_tflops']:.1f} TFLOPS @ {d['best_shape']['m']}x{d['best_shape']['k']}x{d['best_shape']['n']}\")" 2>/dev/null || echo "parse error")
      echo "  ${label}_gpu${gpu}: ${mamf}"
    fi
  done
done
echo "========================================"
