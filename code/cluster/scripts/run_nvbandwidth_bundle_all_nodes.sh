#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_nvbandwidth_bundle_all_nodes.sh --hosts <h1,h2,...> [options]

Runs nvbandwidth bundle on each host and writes:
  results/structured/<run_id>_<label>_nvbandwidth.json
  results/structured/<run_id>_<label>_nvbandwidth_sums.csv
  results/structured/<run_id>_<label>_nvbandwidth_clock_lock.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)

  --runtime <mode>       host|container (default: host)
  --image <image>        Container image for runtime=container, and CUDA compat
                         source image for runtime=host
                         (default: cluster_perf_orig_parity:latest)
  --nvbw-bin <path>      nvbandwidth executable path (default: nvbandwidth)
  --quick                Use reduced testcase subset
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

RUNTIME="${RUNTIME:-host}"
IMAGE="${IMAGE:-cluster_perf_orig_parity:latest}"
NVBW_BIN="${NVBW_BIN:-nvbandwidth}"
QUICK=0

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --nvbw-bin) NVBW_BIN="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if [[ "$RUNTIME" != "host" && "$RUNTIME" != "container" ]]; then
  echo "ERROR: --runtime must be host or container (got: ${RUNTIME})" >&2
  exit 2
fi
IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
LOCAL_STRUCTURED_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
LOCAL_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
REMOTE_STRUCTURED_DIR="$(cluster_structured_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
REMOTE_RAW_DIR="$(cluster_raw_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
REMOTE_ARTIFACT_ENV="$(cluster_artifact_env_prefix_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
mkdir -p "${LOCAL_STRUCTURED_DIR}" "${LOCAL_RAW_DIR}"

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

  out_json_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_nvbandwidth.json"
  out_sums_csv_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_nvbandwidth_sums.csv"
  out_clock_json_remote="${REMOTE_STRUCTURED_DIR}/${RUN_ID}_${label}_nvbandwidth_clock_lock.json"
  out_raw_dir_remote="${REMOTE_RAW_DIR}/${RUN_ID}_${label}_nvbandwidth"

  echo "========================================"
  echo "nvbandwidth: host=${host} label=${label} runtime=${RUNTIME}"
  echo "Outputs: ${out_json_remote}, ${out_sums_csv_remote}, ${out_clock_json_remote}"
  echo "========================================"

  bench_args=(
    scripts/repro/run_nvbandwidth_bundle.sh
    --run-id "${RUN_ID}"
    --label "${label}"
    --runtime "${RUNTIME}"
  )
  if [[ "$RUNTIME" == "container" ]]; then
    bench_args+=(--image "${IMAGE}")
  else
    bench_args+=(--nvbw-bin "${NVBW_BIN}")
    bench_args+=(--compat-image "${IMAGE}")
  fi
  if [[ "$QUICK" -eq 1 ]]; then
    bench_args+=(--quick)
  fi

  bench_str="$(printf '%q ' "${bench_args[@]}")"
  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${REMOTE_ARTIFACT_ENV} ${bench_str}"

  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    bash -lc "$remote_cmd"
  else
    run_remote "$host" "bash -lc $(printf '%q' "$remote_cmd")"
  fi

  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    rm -f "${LOCAL_STRUCTURED_DIR}/$(basename "${out_json_remote}")" "${LOCAL_STRUCTURED_DIR}/$(basename "${out_sums_csv_remote}")" "${LOCAL_STRUCTURED_DIR}/$(basename "${out_clock_json_remote}")"
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_json_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "ERROR: failed to fetch ${out_json_remote} from ${host}" >&2
      exit 1
    }
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_sums_csv_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "ERROR: failed to fetch ${out_sums_csv_remote} from ${host}" >&2
      exit 1
    }
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_clock_json_remote}" "${LOCAL_STRUCTURED_DIR}/" || {
      echo "ERROR: failed to fetch ${out_clock_json_remote} from ${host}" >&2
      exit 1
    }
    rm -rf "${LOCAL_RAW_DIR}/$(basename "${out_raw_dir_remote}")"
    scp -r "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${out_raw_dir_remote}" "${LOCAL_RAW_DIR}/" || {
      echo "ERROR: failed to fetch ${out_raw_dir_remote} from ${host}" >&2
      exit 1
    }
  fi
done

echo "Done."
