#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/collect_software_versions_all_nodes.sh --hosts <h1,h2,...> [options]

Captures a small, reproducible set of software version facts per node into:
  results/structured/${RUN_ID}_${label}_software_versions.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required; hostnames or IPs)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
EOF
}

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
SCRIPT_PATH="${ROOT_DIR}/scripts/collect_software_versions.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
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
OUT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_DIR"

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

is_local_host() {
  local host="$1"
  local hn hn_s
  hn="$(hostname 2>/dev/null || true)"
  hn_s="$(hostname -s 2>/dev/null || true)"
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "::1" ]]; then
    return 0
  fi
  if [[ -n "$hn" && "$host" == "$hn" ]]; then
    return 0
  fi
  if [[ -n "$hn_s" && "$host" == "$hn_s" ]]; then
    return 0
  fi
  return 1
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

  out_file="${RUN_ID}_${label}_software_versions.json"
  echo "=== ${label} (${host}) ==="

  if is_local_host "$host"; then
    bash "${SCRIPT_PATH}" --output "${OUT_DIR}/${out_file}" --label "${label}"
    echo "Saved ${OUT_DIR}/${out_file}"
    continue
  fi

  tmp_dir="/tmp/cluster_sw_versions_${RUN_ID}"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "mkdir -p ${tmp_dir}" || { echo "SSH failed for ${host}" >&2; continue; }
  scp "${SSH_OPTS[@]}" "$SCRIPT_PATH" "${SSH_USER}@${host}:${tmp_dir}/" || { echo "SCP failed for ${host}" >&2; continue; }
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash ${tmp_dir}/collect_software_versions.sh --venv-py \\$HOME/ai-performance-engineering/code/cluster/env/venv/bin/python --output ${tmp_dir}/${out_file} --label ${label}" || { echo "Remote collection failed for ${host}" >&2; continue; }
  scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${tmp_dir}/${out_file}" "${OUT_DIR}/" || { echo "Fetch failed for ${host}" >&2; continue; }

  echo "Saved ${OUT_DIR}/${out_file}"
done
