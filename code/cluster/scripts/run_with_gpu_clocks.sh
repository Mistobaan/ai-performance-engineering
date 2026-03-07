#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_with_gpu_clocks.sh [--devices <i,j,k>] [--lock-meta-out <path>] -- <command> [args...]

Notes:
  - This locks clocks via the repo harness `lock_gpu_clocks`.
  - You do NOT need to run this script as root; `lock_gpu_clocks` will use
    passwordless `sudo -n` for `nvidia-smi` when needed.
  - Clock locking is REQUIRED. This script fails if clock lock cannot be acquired.
EOF
}

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"
VENV_PY="${VENV_PY:-${ROOT_DIR}/env/venv/bin/python}"

DEVICES=""
LOCK_META_OUT=""

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

# Accept the common "-- <cmd>" delimiter pattern.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --devices)
      DEVICES="${2:-}"
      shift 2
      ;;
    --lock-meta-out)
      LOCK_META_OUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: venv python not found or not executable at $VENV_PY" >&2
  exit 1
fi

if [[ -z "$LOCK_META_OUT" ]]; then
  # If the caller provides RUN_ID/LABEL (common in this repo), default to a
  # structured lock metadata file so clock-lock evidence is always retained.
  if [[ -n "${RUN_ID:-}" ]]; then
    out_dir="$(cluster_structured_dir_for_root "${ROOT_DIR}" "${RUN_ID}")"
    mkdir -p "$out_dir"
    if [[ -n "${LABEL:-}" ]]; then
      LOCK_META_OUT="${out_dir}/${RUN_ID}_${LABEL}_clock_lock.json"
    else
      LOCK_META_OUT="${out_dir}/${RUN_ID}_clock_lock.json"
    fi
  fi
fi

args=()
if [[ -n "$DEVICES" ]]; then
  args+=(--devices "$DEVICES")
fi
if [[ -n "$LOCK_META_OUT" ]]; then
  args+=(--lock-meta-out "$LOCK_META_OUT")
fi

"$VENV_PY" "${ROOT_DIR}/scripts/lock_and_run.py" "${args[@]}" -- "$@"
