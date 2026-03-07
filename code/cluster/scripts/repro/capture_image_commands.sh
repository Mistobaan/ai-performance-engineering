#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Capture the terminal commands shown in the case-study screenshots (IMEX status,
ipinfo geo, nvidia-smi, dcgmi discovery, topo).

Usage:
  scripts/repro/capture_image_commands.sh [--run-id <id>] [--hosts <h1,h2,...>] [--ssh-key <path>]

Notes:
- If --hosts is omitted, captures locally only.
- Remote capture uses SSH: ubuntu@<host>.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
SSH_KEY="${SSH_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --hosts)
      HOSTS="$2"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR"

sanitize_label() {
  # Safe filename label: keep alnum, dot, dash, underscore; replace others with underscore.
  echo "$1" | tr -c '[:alnum:]._-' '_'
}

run_local() {
  local cmd="$1"
  echo "\$ ${cmd}"
  bash -lc "$cmd" 2>&1 || true
  echo
}

run_remote() {
  local host="$1"
  local cmd="$2"

  local -a ssh_opts=(
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=10
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
  )
  if [[ -n "$SSH_KEY" ]]; then
    ssh_opts+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
  fi

  echo "\$ ssh ${host} '${cmd}'"
  ssh "${ssh_opts[@]}" "ubuntu@${host}" "bash -lc $(printf '%q' "$cmd")" 2>&1 || true
  echo
}

COMMANDS=(
  "date"
  "hostname"
  "uname -a"
  "systemctl status nvidia-imex --no-pager"
  "command -v nvidia-imex-ctl >/dev/null 2>&1 && nvidia-imex-ctl -q || echo 'nvidia-imex-ctl not found'"
  "systemctl status nvidia-fabricmanager --no-pager"
  "nvidia-smi"
  "nvidia-smi topo -m"
  "dcgmi discovery -l"
  "if command -v jq >/dev/null 2>&1; then curl -s https://ipinfo.io | jq .; else curl -s https://ipinfo.io; fi"
)

HOST_ARR=()
if [[ -n "$HOSTS" ]]; then
  IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
else
  HOST_ARR=("local")
fi

LOCAL_HOSTNAME="$(hostname)"
for host in "${HOST_ARR[@]}"; do
  host="${host//[[:space:]]/}"
  if [[ -z "$host" ]]; then
    continue
  fi

  label="$host"
  if [[ "$host" == "local" ]]; then
    label="$LOCAL_HOSTNAME"
  fi

  safe_label="$(sanitize_label "$label")"
  out_path="${OUT_RAW_DIR}/${RUN_ID}_${safe_label}_image_cmds.log"

  {
    echo "# $(date -Is)"
    echo "# label: ${label}"
    echo
    for cmd in "${COMMANDS[@]}"; do
      if [[ "$host" == "local" ]]; then
        run_local "$cmd"
      else
        run_remote "$host" "$cmd"
      fi
    done
  } | tee "$out_path" >/dev/null

  echo "Wrote ${out_path}"
done
