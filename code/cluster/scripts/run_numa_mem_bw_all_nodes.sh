#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_numa_mem_bw_all_nodes.sh --hosts <h1,h2,...> [options]

Runs NUMA memory bandwidth probe on each host (see run_numa_mem_bw.sh) and
fetches the structured artifacts back to the driver:
  runs/<run_id>/structured/<run_id>_<label>_numa_mem_bw.json
  runs/<run_id>/structured/<run_id>_<label>_numa_mem_bw.csv

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)

  --bytes <n>            Bytes per memcpy iteration (default: 1073741824)
  --iters <n>            Iterations (default: 10)
  --threads <n>          Threads (default: 16)
  --warmup <n>           Warmup iterations (default: 2)
  --nodes <csv>          Explicit NUMA nodes to test (default: auto)
  --cpu-node <n>         CPU node to run on (default: first CPU NUMA node)
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

BYTES="${BYTES:-1073741824}"
ITERS="${ITERS:-10}"
THREADS="${THREADS:-16}"
WARMUP="${WARMUP:-2}"
NODES="${NODES:-}"
CPU_NODE="${CPU_NODE:-}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --cpu-node) CPU_NODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
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

  out_json="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_numa_mem_bw.json"
  out_csv="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_numa_mem_bw.csv"

  echo "========================================"
  echo "NUMA mem BW: host=${host} label=${label}"
  echo "Output: ${out_json}"
  echo "========================================"

  bench_args=(
    scripts/run_numa_mem_bw.sh
    --run-id "${RUN_ID}"
    --label "${label}"
    --bytes "${BYTES}"
    --iters "${ITERS}"
    --threads "${THREADS}"
    --warmup "${WARMUP}"
  )
  if [[ -n "$NODES" ]]; then
    bench_args+=(--nodes "${NODES}")
  fi
  if [[ -n "$CPU_NODE" ]]; then
    bench_args+=(--cpu-node "${CPU_NODE}")
  fi

  bench_str="$(printf '%q ' "${bench_args[@]}")"
  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${bench_str}"

  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    bash -lc "$remote_cmd"
  else
    run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")"
  fi

  # Fetch results back to the driver for plotting/reporting.
  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_json}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_json} from ${host}" >&2
    }
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_csv}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_csv} from ${host}" >&2
    }
  fi
done

echo "Done."
