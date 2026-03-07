#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_checkpoint_io_all_nodes.sh --hosts <h1,h2,...> [options]

Runs a checkpoint-like I/O benchmark (write + read) on each host and writes:
  runs/<run_id>/structured/<run_id>_<label>_checkpoint_io.json
  runs/<run_id>/structured/<run_id>_<label>_checkpoint_io.csv

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)

  --test-dir <path>      Directory to write files under (default: /tmp)
  --bytes <size>         Bytes per checkpoint file (default: 4G)
  --block-size <size>    Block size (default: 4M)
  --files <n>            Number of files (default: 1)
  --fsync <0|1>          fsync after write (default: 1)
  --write <0|1>          Run write test (default: 1)
  --read <0|1>           Run read test (default: 1)
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

TEST_DIR="${TEST_DIR:-/tmp}"
BYTES="${BYTES:-4G}"
BLOCK_SIZE="${BLOCK_SIZE:-4M}"
FILES="${FILES:-1}"
FSYNC="${FSYNC:-1}"
DO_WRITE="${DO_WRITE:-1}"
DO_READ="${DO_READ:-1}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;

    --test-dir) TEST_DIR="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --files) FILES="$2"; shift 2 ;;
    --fsync) FSYNC="$2"; shift 2 ;;
    --write) DO_WRITE="$2"; shift 2 ;;
    --read) DO_READ="$2"; shift 2 ;;
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

  out_json="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_checkpoint_io.json"
  out_csv="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_checkpoint_io.csv"

  echo "========================================"
  echo "Checkpoint I/O: host=${host} label=${label}"
  echo "Output: ${out_json}"
  echo "========================================"

  bench_args=(
    env/venv/bin/python
    scripts/checkpoint_io_bench.py
    --run-id "${RUN_ID}"
    --label "${label}"
    --test-dir "${TEST_DIR}"
    --bytes "${BYTES}"
    --block-size "${BLOCK_SIZE}"
    --files "${FILES}"
    --fsync "${FSYNC}"
    --write "${DO_WRITE}"
    --read "${DO_READ}"
    --output-json "${out_json}"
    --output-csv "${out_csv}"
  )

  bench_str="$(printf '%q ' "${bench_args[@]}")"
  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && RUN_ID=$(printf '%q' "${RUN_ID}") LABEL=$(printf '%q' "${label}") TEST_DIR=$(printf '%q' "${TEST_DIR}") BYTES=$(printf '%q' "${BYTES}") BLOCK_SIZE=$(printf '%q' "${BLOCK_SIZE}") FILES=$(printf '%q' "${FILES}") FSYNC=$(printf '%q' "${FSYNC}") DO_WRITE=$(printf '%q' "${DO_WRITE}") DO_READ=$(printf '%q' "${DO_READ}") ${bench_str}"

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
