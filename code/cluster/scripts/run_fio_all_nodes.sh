#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_fio_all_nodes.sh --hosts <h1,h2,...> [options]

Runs fio storage benchmarks on each host and writes:
  results/structured/<run_id>_<label>_fio.json
  results/structured/<run_id>_<label>_fio.csv
  results/structured/<run_id>_<label>_fio_stability.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)

  --test-dir <path>      fio test directory (default: /tmp)
  --file-size <size>     fio size per job (default: 8G)
  --runtime <sec>        fio runtime seconds (default: 60)
  --repeats <n>          fio repetitions per host (default: 1)
  --numjobs <n>          fio numjobs (default: 4)
  --iodepth <n>          fio iodepth for random tests (default: 64)
  --bs-seq <size>        fio block size for seq tests (default: 1M)
  --bs-rand <size>       fio block size for random tests (default: 4K)
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
FILE_SIZE="${FILE_SIZE:-8G}"
RUNTIME="${RUNTIME:-60}"
REPEATS="${REPEATS:-1}"
NUMJOBS="${NUMJOBS:-4}"
IODEPTH="${IODEPTH:-64}"
BS_SEQ="${BS_SEQ:-1M}"
BS_RAND="${BS_RAND:-4K}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --test-dir) TEST_DIR="$2"; shift 2 ;;
    --file-size) FILE_SIZE="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --numjobs) NUMJOBS="$2"; shift 2 ;;
    --iodepth) IODEPTH="$2"; shift 2 ;;
    --bs-seq) BS_SEQ="$2"; shift 2 ;;
    --bs-rand) BS_RAND="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --repeats must be a positive integer (got: ${REPEATS})" >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
LOCAL_STRUCTURED_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
LOCAL_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
REMOTE_STRUCTURED_DIR="$(cluster_structured_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
REMOTE_RAW_DIR="$(cluster_raw_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
REMOTE_ARTIFACT_ENV="$(cluster_artifact_env_prefix_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
mkdir -p "${LOCAL_STRUCTURED_DIR}" "${LOCAL_RAW_DIR}"

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

  out_json_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_fio.json"
  out_csv_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_fio.csv"
  out_stability_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_fio_stability.json"
  out_raw_dir_remote="${REMOTE_RAW_DIR}/${RUN_ID}_${label}_fio"

  echo "========================================"
  echo "fio: host=${host} label=${label}"
  echo "Outputs: ${out_json_remote}, ${out_csv_remote}, ${out_stability_remote}"
  echo "========================================"

  bench_args=(
    scripts/run_fio_bench.sh
    --run-id "${RUN_ID}"
    --label "${label}"
    --test-dir "${TEST_DIR}"
    --file-size "${FILE_SIZE}"
    --runtime "${RUNTIME}"
    --repeats "${REPEATS}"
    --numjobs "${NUMJOBS}"
    --iodepth "${IODEPTH}"
    --bs-seq "${BS_SEQ}"
    --bs-rand "${BS_RAND}"
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
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_stability_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "WARNING: failed to fetch ${out_stability_remote} from ${host}" >&2
    }
    rm -rf "${LOCAL_RAW_DIR}/${RUN_ID}_${label}_fio"
    scp -r "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_raw_dir_remote}" "${LOCAL_RAW_DIR}/" || {
      echo "WARNING: failed to fetch ${out_raw_dir_remote} from ${host}" >&2
    }
  fi
done

echo "Done."
