#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_gemm_with_telemetry_all_nodes.sh --hosts <h1,h2,...> [options]

Runs `scripts/run_gemm_with_telemetry.sh` on each host (via ssh when needed) and
fetches artifacts back to the driver node (this machine) for plotting/reporting.

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD_HHMMSS_gemm_telemetry)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)
  --gpus <i,j,k>         Comma-separated physical GPU indices (default: 0)
  --m <int>              GEMM M (default: 16384)
  --n <int>              GEMM N (default: 16384)
  --k <int>              GEMM K (default: 16384)
  --dtype <bf16|fp16>    Data type (default: bf16)
  --iters <n>            Iterations (default: 10000)
  --interval <sec>       Telemetry sampling interval seconds (default: 1)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H%M%S)_gemm_telemetry}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"
GPUS="0"

M="${M:-16384}"
N="${N:-16384}"
K="${K:-16384}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-10000}"
INTERVAL="${INTERVAL:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
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

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage
  exit 2
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
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
LOCAL_STRUCTURED_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
LOCAL_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
REMOTE_STRUCTURED_DIR="$(cluster_structured_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
REMOTE_RAW_DIR="$(cluster_raw_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
mkdir -p "$LOCAL_STRUCTURED_DIR" "$LOCAL_RAW_DIR"

run_remote() {
  local host="$1"
  local cmd="$2"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "$cmd")"
}

fetch_remote() {
  local host="$1"
  local path="$2"
  local dest_dir="$3"
  mkdir -p "$dest_dir"
  scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${REMOTE_ROOT}/${path}" "${dest_dir}/" || {
    echo "WARNING: failed to fetch ${path} from ${host}" >&2
    return 1
  }
}

IFS=',' read -r -a GPU_ARR <<<"$GPUS"

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
  echo "GEMM+telemetry: host=${host} label=${label}"
  echo "RUN_ID=${RUN_ID}"
  echo "GPUS=${GPUS}"
  echo "========================================"

  for gpu in "${GPU_ARR[@]}"; do
    gpu="$(echo "$gpu" | xargs)"
    [[ -n "$gpu" ]] || continue

    cmd="cd $(printf '%q' "${REMOTE_ROOT}") && RUN_ID=$(printf '%q' "${RUN_ID}") LABEL=$(printf '%q' "${label}") scripts/run_gemm_with_telemetry.sh --gpu $(printf '%q' "${gpu}") --m $(printf '%q' "${M}") --n $(printf '%q' "${N}") --k $(printf '%q' "${K}") --dtype $(printf '%q' "${DTYPE}") --iters $(printf '%q' "${ITERS}") --interval $(printf '%q' "${INTERVAL}")"
    if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
      bash -lc "$cmd"
    else
      run_remote "$host" "$cmd"
      # Fetch artifacts back to the driver node.
      fetch_remote "$host" "${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_gemm.csv" "${LOCAL_STRUCTURED_DIR}" || true
      fetch_remote "$host" "${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu${gpu}_gemm_clock_lock.json" "${LOCAL_STRUCTURED_DIR}" || true
      fetch_remote "$host" "${REMOTE_RAW_DIR}/${RUN_ID}_${label}_gpu${gpu}_gemm.log" "${LOCAL_RAW_DIR}" || true
      fetch_remote "$host" "${REMOTE_RAW_DIR}/${RUN_ID}_${label}_gpu${gpu}_gemm_telemetry_query.csv" "${LOCAL_RAW_DIR}" || true
      fetch_remote "$host" "${REMOTE_RAW_DIR}/${RUN_ID}_${label}_gpu${gpu}_gemm_telemetry_pmon.log" "${LOCAL_RAW_DIR}" || true
    fi
  done
done

echo "Done."
