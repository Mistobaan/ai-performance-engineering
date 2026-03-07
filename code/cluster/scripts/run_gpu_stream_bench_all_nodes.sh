#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_gpu_stream_bench_all_nodes.sh --hosts <h1,h2,...> [options]

Runs GPU STREAM-style benchmark on each host and writes:
  results/structured/<run_id>_<label>_gpu_stream.json
  results/structured/<run_id>_<label>_gpu_stream.csv
  results/structured/<run_id>_<label>_gpu_stream_clock_lock.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)

  --device <n>           CUDA device index (default: 0)
  --size-mb <n>          Vector size in MB (default: 1024)
  --iters <n>            Measured iterations per op (default: 40)
  --warmup <n>           Warmup iterations per op (default: 10)
  --dtype <fp32|fp16|bf16>  Data type (default: fp32)
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

DEVICE="${DEVICE:-0}"
SIZE_MB="${SIZE_MB:-1024}"
ITERS="${ITERS:-40}"
WARMUP="${WARMUP:-10}"
DTYPE="${DTYPE:-fp32}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --size-mb) SIZE_MB="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
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

  out_json_local="${LOCAL_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream.json"
  out_csv_local="${LOCAL_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream.csv"
  out_lock_local="${LOCAL_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream_clock_lock.json"
  out_json_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream.json"
  out_csv_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream.csv"
  out_lock_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_gpu_stream_clock_lock.json"

  echo "========================================"
  echo "gpu_stream: host=${host} label=${label}"
  echo "Outputs: ${out_json_remote}, ${out_csv_remote}, ${out_lock_remote}"
  echo "========================================"

  bench_args=(
    scripts/run_gpu_stream_bench.sh
    --run-id "${RUN_ID}"
    --label "${label}"
    --device "${DEVICE}"
    --size-mb "${SIZE_MB}"
    --iters "${ITERS}"
    --warmup "${WARMUP}"
    --dtype "${DTYPE}"
  )
  bench_str="$(printf '%q ' "${bench_args[@]}")"
  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${REMOTE_ARTIFACT_ENV} ${bench_str}"

  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    bash -lc "$remote_cmd"
  else
    run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")"
  fi

  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_json_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_json_remote} from ${host}" >&2
    }
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_csv_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_csv_remote} from ${host}" >&2
    }
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_lock_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_lock_remote} from ${host}" >&2
    }
  fi
done

echo "Done."
