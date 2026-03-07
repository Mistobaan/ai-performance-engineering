#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_gemm_sanity_all_nodes.sh --hosts <h1,h2,...> [options]

Runs a per-GPU Torch GEMM sanity check on each host to quickly detect
per-node/per-GPU throughput deltas.

With --concurrent, all GPUs on each node run simultaneously, which
detects "straggler" GPUs (the slowest GPU sets training pace for all).
The summary reports min/max spread and flags stragglers (>2% slower).

Outputs (per host):
  results/structured/<run_id>_<label>_gemm_gpu_sanity.csv

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs on driver)
  --m <int>              GEMM M (default: 16384)
  --n <int>              GEMM N (default: 16384)
  --k <int>              GEMM K (default: 16384)
  --dtype <bf16|fp16>    Data type (default: bf16)
  --iters <n>            Iterations per GPU (default: 50)
  --concurrent           Run all GPUs simultaneously (straggler detection)
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

M="${M:-16384}"
N="${N:-16384}"
K="${K:-16384}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-50}"
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
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
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

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
LOCAL_STRUCTURED_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
REMOTE_STRUCTURED_DIR="$(cluster_structured_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
REMOTE_ARTIFACT_ENV="$(cluster_artifact_env_prefix_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
mkdir -p "${LOCAL_STRUCTURED_DIR}"

if [[ -z "$GPUS_PER_NODE" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot auto-detect --gpus-per-node" >&2
    exit 2
  fi
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

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

  out_csv_local="${LOCAL_STRUCTURED_DIR}/${RUN_ID}_${label}_gemm_gpu_sanity.csv"
  out_csv_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gemm_gpu_sanity.csv"

  mode_label="sequential"
  if $CONCURRENT; then
    mode_label="concurrent (straggler detection)"
  fi

  echo "========================================"
  echo "GEMM sanity: host=${host} label=${label} mode=${mode_label}"
  echo "Output: ${out_csv_remote}"
  echo "========================================"

  if $CONCURRENT; then
    # Concurrent: launch all GPUs simultaneously (straggler detection)
    pids=()
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
      echo "  Launching GPU${gpu} (concurrent)..."
      bench_args=(
        scripts/run_with_gpu_clocks.sh
        --devices "${gpu}"
        --
        env/venv/bin/python
        scripts/torch_gemm_bench.py
        --m "${M}"
        --n "${N}"
        --k "${K}"
        --dtype "${DTYPE}"
        --iters "${ITERS}"
        --label "${label}_gpu${gpu}"
        --output-csv "${out_csv_remote}"
      )
      bench_str="$(printf '%q ' "${bench_args[@]}")"
      remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${REMOTE_ARTIFACT_ENV} CUDA_VISIBLE_DEVICES=${gpu} RUN_ID=$(printf '%q' "${RUN_ID}") LABEL=$(printf '%q' "${label}_gpu${gpu}") ${bench_str}"
      if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
        bash -lc "$remote_cmd" &
        pids+=($!)
      else
        run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")" &
        pids+=($!)
      fi
    done
    echo "  Waiting for ${#pids[@]} concurrent GPU runs..."
    for pid in "${pids[@]}"; do
      wait "$pid" || echo "WARNING: PID $pid exited non-zero" >&2
    done
    echo "  All GPUs complete for ${label}."
  else
    # Sequential: one GPU at a time (original behavior)
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
      echo "== ${label} GPU${gpu} =="
      bench_args=(
        scripts/run_with_gpu_clocks.sh
        --
        env/venv/bin/python
        scripts/torch_gemm_bench.py
        --m "${M}"
        --n "${N}"
        --k "${K}"
        --dtype "${DTYPE}"
        --iters "${ITERS}"
        --label "${label}_gpu${gpu}"
        --output-csv "${out_csv_remote}"
      )
      bench_str="$(printf '%q ' "${bench_args[@]}")"
      remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${REMOTE_ARTIFACT_ENV} CUDA_VISIBLE_DEVICES=${gpu} RUN_ID=$(printf '%q' "${RUN_ID}") LABEL=$(printf '%q' "${label}_gpu${gpu}") ${bench_str}"
      if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
        bash -lc "$remote_cmd"
      else
        # IMPORTANT: keep the -c payload as a single shell token on the remote
        # side; otherwise ssh will split it and `bash -lc` will only receive the
        # first word ("cd"), causing the rest to execute in the wrong directory.
        run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")"
      fi
    done
  fi

  # Fetch results back to the driver for plotting/reporting.
  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_csv_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_csv_remote} from ${host}" >&2
    }
  fi
done

# Straggler detection summary (when --concurrent is used)
if $CONCURRENT; then
  echo ""
  echo "========================================"
  echo "GPU Straggler Detection Summary"
  echo "========================================"
  for idx in "${!HOST_ARR[@]}"; do
    host="$(echo "${HOST_ARR[$idx]}" | xargs)"
    [[ -n "$host" ]] || continue
    slabel=""
    if [[ -n "$LABELS" ]]; then
      slabel="$(echo "${LABEL_ARR[$idx]}" | xargs)"
    fi
    if [[ -z "$slabel" ]]; then
      slabel="$(sanitize_label "$host")"
    fi
    scsv="${LOCAL_STRUCTURED_DIR}/${RUN_ID}_${slabel}_gemm_gpu_sanity.csv"
    if [[ -f "${scsv}" ]]; then
      if ! python3 - "${scsv}" "${slabel}" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print(f"  {sys.argv[2]}: no data"); sys.exit(0)
tflops = [(r.get('label',''), float(r.get('avg_tflops', 0))) for r in rows if r.get('avg_tflops')]
if not tflops:
    print(f"  {sys.argv[2]}: no TFLOPS data"); sys.exit(0)
tflops.sort(key=lambda x: x[1])
mn_l, mn_v = tflops[0]; mx_l, mx_v = tflops[-1]
sp = (mx_v - mn_v) / mx_v * 100 if mx_v > 0 else 0
print(f"  {sys.argv[2]}: min={mn_v:.1f} ({mn_l}), max={mx_v:.1f} ({mx_l}), spread={sp:.1f}%")
if sp > 2:
    print(f"    WARNING: {mn_l} is a straggler ({sp:.1f}% slower than fastest)")
else:
    print(f"    OK: all GPUs within 2% of each other")
PYEOF
      then
        echo "  ${slabel}: parse error" >&2
      fi
    fi
  done
  echo "========================================"
fi

echo "Done."
